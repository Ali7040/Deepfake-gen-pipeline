"use client";

import { useEffect, useState } from "react";
import RequireAuth from "@/components/auth/RequireAuth";
import AppHeader from "@/components/layout/AppHeader";
import MagneticButton from "@/components/ui/MagneticButton";
import GeneratingLoader from "@/components/ui/GeneratingLoader";
import VideoProgressLoader from "@/components/ui/VideoProgressLoader";
import LoadingDots from "@/components/ui/LoadingDots";
import {
  genSwap,
  genDetectFaces,
  swapOutputSrc,
  pollSwapJob,
  friendlyErrorMessage,
  type FaceCrop,
  type SwapProgress,
} from "@/lib/api";
import clsx from "clsx";

function UploadTile({
  label,
  file,
  onPick,
  acceptVideo,
}: {
  label: string;
  file: File | null;
  onPick: (f: File) => void;
  acceptVideo?: boolean;
}) {
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [mediaReady, setMediaReady] = useState(false);
  const [previewFailed, setPreviewFailed] = useState(false);
  const [uploading, setUploading] = useState(false);

  const isVideo =
    file?.type.startsWith("video/") ||
    /\.(mp4|mov|avi|mkv|webm|m4v)$/i.test(file?.name ?? "");

  // File-type detection (MIME sniffing, extension matching) can be wrong for
  // odd encodings, and a bad/unsupported preview must never hang the UI
  // forever waiting on an event that will never fire — always clear the
  // loading state after a hard cap regardless of what happened.
  useEffect(() => {
    if (!previewUrl) return;
    setMediaReady(false);
    setPreviewFailed(false);
    const timeout = window.setTimeout(() => setMediaReady(true), 6000);
    return () => window.clearTimeout(timeout);
  }, [previewUrl]);

  return (
    <label className="glass relative flex aspect-square cursor-pointer flex-col items-center justify-center overflow-hidden rounded-2xl border-2 border-dashed border-white/15 p-6 text-center transition-colors hover:border-accent-blue">
      {previewUrl && isVideo && !previewFailed ? (
        <video
          src={previewUrl}
          muted
          loop
          autoPlay
          playsInline
          onLoadedData={() => setMediaReady(true)}
          onError={() => {
            setMediaReady(true);
            setPreviewFailed(true);
          }}
          className={clsx(
            "max-h-full max-w-full rounded-lg object-contain transition-opacity",
            mediaReady ? "opacity-100" : "opacity-0"
          )}
        />
      ) : previewUrl && !isVideo && !previewFailed ? (
        // eslint-disable-next-line @next/next/no-img-element
        <img
          src={previewUrl ?? undefined}
          alt={label}
          onLoad={() => setMediaReady(true)}
          onError={() => {
            setMediaReady(true);
            setPreviewFailed(true);
          }}
          className={clsx(
            "max-h-full max-w-full rounded-lg object-contain transition-opacity",
            mediaReady ? "opacity-100" : "opacity-0"
          )}
        />
      ) : previewFailed && file ? (
        <div className="flex flex-col items-center gap-1 px-4">
          <p className="text-sm text-ink-dim">{file.name}</p>
          <p className="font-mono text-xs text-ink-faint">
            Preview unavailable, but the file is ready to submit.
          </p>
        </div>
      ) : (
        <>
          <p className="text-sm text-ink-dim">{label}</p>
          <p className="mt-1 font-mono text-xs text-ink-faint">
            Click to browse{acceptVideo ? " — image or video" : ""}
          </p>
        </>
      )}

      {previewUrl && !mediaReady && (
        <div className="absolute inset-0 flex items-center justify-center bg-void/60">
          <LoadingDots size="lg" className="text-accent-blue" />
        </div>
      )}
      {uploading && (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 bg-void/70">
          <LoadingDots size="lg" className="text-accent-blue" />
          <p className="font-mono text-xs uppercase tracking-widest text-accent-blue">
            Uploading{isVideo ? " video" : ""}…
          </p>
        </div>
      )}

      <input
        type="file"
        accept={acceptVideo ? "image/*,video/*" : "image/*"}
        className="hidden"
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (!f) return;
          setUploading(true);
          onPick(f);
          setPreviewUrl(URL.createObjectURL(f));
          // Reading a large file (esp. video) into a blob URL can take a
          // beat; give the user a brief, honest "uploading" state rather
          // than an instant, jarring preview pop-in.
          const delay = f.size > 8 * 1024 * 1024 ? 900 : 350;
          window.setTimeout(() => setUploading(false), delay);
        }}
      />
    </label>
  );
}

function FacePicker({
  faces,
  selected,
  onToggle,
}: {
  faces: FaceCrop[];
  selected: Set<number>;
  onToggle: (i: number) => void;
}) {
  if (faces.length <= 1) return null;

  return (
    <div className="mt-4">
      <p className="font-mono text-xs uppercase tracking-widest text-ink-dim">
        {faces.length} faces found — choose which to swap (none selected = all)
      </p>
      <div className="mt-3 flex flex-wrap gap-3">
        {faces.map((face, i) => (
          <button
            key={i}
            type="button"
            data-cursor-hover
            onClick={() => onToggle(i)}
            className={clsx(
              "relative h-16 w-16 overflow-hidden rounded-lg border-2 transition-colors",
              selected.has(i) ? "border-accent-blue" : "border-white/15"
            )}
          >
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={`data:image/jpeg;base64,${face.b64}`}
              alt={`Face ${i + 1}`}
              className="h-full w-full object-cover"
            />
            {selected.has(i) && (
              <span className="absolute inset-0 flex items-center justify-center bg-accent-blue/30 text-xs font-bold text-white">
                ✓
              </span>
            )}
          </button>
        ))}
      </div>
    </div>
  );
}

function GeneratePageContent() {
  const [source, setSource] = useState<File | null>(null);
  const [target, setTarget] = useState<File | null>(null);
  const [detectedFaces, setDetectedFaces] = useState<FaceCrop[]>([]);
  const [selectedFaces, setSelectedFaces] = useState<Set<number>>(new Set());
  const [output, setOutput] = useState<{ type: "image" | "video"; src: string } | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [videoProgress, setVideoProgress] = useState<SwapProgress | null>(null);
  const [enhanceVideo, setEnhanceVideo] = useState(false);

  const isVideoTarget = target?.type.startsWith("video/") ?? false;

  useEffect(() => {
    if (!target || target.type.startsWith("video/")) {
      setDetectedFaces([]);
      setSelectedFaces(new Set());
      return;
    }
    let cancelled = false;
    genDetectFaces(target)
      .then((res) => {
        if (!cancelled) setDetectedFaces(res.faces ?? []);
      })
      .catch(() => {
        if (!cancelled) setDetectedFaces([]);
      });
    return () => {
      cancelled = true;
    };
  }, [target]);

  const toggleFace = (i: number) => {
    setSelectedFaces((prev) => {
      const next = new Set(prev);
      if (next.has(i)) next.delete(i);
      else next.add(i);
      return next;
    });
  };

  const onSubmit = async () => {
    if (!source || !target) return;
    setLoading(true);
    setError(null);
    setOutput(null);
    setVideoProgress(null);
    try {
      const face_indices =
        selectedFaces.size > 0 ? JSON.stringify(Array.from(selectedFaces)) : "";

      let res;
      if (isVideoTarget) {
        // Video swaps are frame-by-frame and CPU-bound — can run well past a
        // minute, and CodeFormer enhancement roughly doubles that. Run async
        // and poll real per-frame progress instead of blocking on one
        // request with no feedback; enhancement is opt-in for video (off by
        // default) since it's the single biggest cost per frame.
        const started = await genSwap(source, target, {
          face_indices,
          async_mode: true,
          enhance: enhanceVideo ? "1" : "0",
        });
        if (!started.success || !started.job_id) {
          setError(started.error || "Could not start generation.");
          return;
        }
        res = await pollSwapJob(started.job_id, setVideoProgress);
      } else {
        res = await genSwap(source, target, { face_indices });
      }

      if (!res.success) {
        setError(res.error || "Generation failed.");
        return;
      }
      const resolved = swapOutputSrc(res);
      setOutput(resolved);
      if (!resolved) {
        setError(
          "Swap completed but no output could be resolved — check the generation engine response shape."
        );
      }
    } catch (err) {
      setError(friendlyErrorMessage(err));
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-void pb-32">
      <AppHeader />

      <div className="mx-auto max-w-5xl px-6 py-16">
        <p className="font-mono text-xs uppercase tracking-[0.2em] text-accent-orange">
          Generation Engine
        </p>
        <h1 className="mt-3 font-display text-4xl font-medium tracking-tight text-ink sm:text-5xl">
          Generate a face swap
        </h1>
        <p className="mt-4 max-w-xl text-ink-dim">
          Upload a source face and a target image or video. DeepTrace runs the
          swap pipeline and returns the generated result.
        </p>

        <div className="mt-12 grid gap-6 sm:grid-cols-2">
          <UploadTile label="Source face" file={source} onPick={setSource} />
          <UploadTile label="Target image or video" file={target} onPick={setTarget} acceptVideo />
        </div>

        <FacePicker faces={detectedFaces} selected={selectedFaces} onToggle={toggleFace} />

        {isVideoTarget && (
          <label className="mt-4 flex cursor-pointer items-center gap-2 text-sm text-ink-dim">
            <input
              type="checkbox"
              checked={enhanceVideo}
              onChange={(e) => setEnhanceVideo(e.target.checked)}
              className="h-4 w-4 accent-accent-blue"
            />
            Enhance faces (higher quality, roughly doubles processing time)
          </label>
        )}

        <MagneticButton
          className="mt-6 w-full justify-center sm:w-auto"
          onClick={onSubmit}
        >
          {loading ? (
            <span className="inline-flex items-center gap-2">
              Generating
              <LoadingDots size="sm" />
            </span>
          ) : (
            "Run Generation"
          )}
        </MagneticButton>

        {error && (
          <p className="mt-4 rounded-lg border border-accent-orange/30 bg-accent-orange/10 px-3 py-2 text-sm text-accent-orange">
            {error}
          </p>
        )}

        <div className="glass mt-10 rounded-2xl p-6">
          <p className="font-mono text-xs uppercase tracking-widest text-ink-dim">
            Output
          </p>

          {loading && isVideoTarget && <VideoProgressLoader progress={videoProgress} />}
          {loading && !isVideoTarget && <GeneratingLoader />}

          {!loading && !output && (
            <p className="mt-4 text-sm text-ink-faint">
              Your generated result will appear here.
            </p>
          )}

          {!loading && output && output.type === "video" && (
            <video
              src={output.src}
              controls
              autoPlay
              loop
              className="mt-4 w-full rounded-lg"
            />
          )}

          {!loading && output && output.type === "image" && (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={output.src}
              alt="Generated output"
              className="mt-4 w-full rounded-lg"
              onError={() =>
                setError(
                  "The output image failed to load from the server — the file may not have been saved correctly."
                )
              }
            />
          )}
        </div>
      </div>
    </main>
  );
}

export default function GeneratePage() {
  return (
    <RequireAuth>
      <GeneratePageContent />
    </RequireAuth>
  );
}
