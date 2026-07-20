"use client";

import { useRef, useState } from "react";
import RequireAuth from "@/components/auth/RequireAuth";
import AppHeader from "@/components/layout/AppHeader";
import MagneticButton from "@/components/ui/MagneticButton";
import { detectImage, mediaUrl, ApiError, type ImageDetectionResult } from "@/lib/api";
import LoadingDots from "@/components/ui/LoadingDots";
import clsx from "clsx";

function DetectPageContent() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [result, setResult] = useState<ImageDetectionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const [mediaReady, setMediaReady] = useState(false);
  const [uploading, setUploading] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const pickFile = (f: File) => {
    setFile(f);
    setMediaReady(false);
    setUploading(true);
    setPreviewUrl(URL.createObjectURL(f));
    setResult(null);
    setError(null);
    const delay = f.size > 8 * 1024 * 1024 ? 900 : 350;
    window.setTimeout(() => setUploading(false), delay);
  };

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragActive(false);
    const f = e.dataTransfer.files?.[0];
    if (f) pickFile(f);
  };

  const onSubmit = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const res = await detectImage(file);
      setResult(res);
    } catch (err) {
      setError(
        err instanceof ApiError
          ? err.message
          : "Could not reach the detection engine. Is the backend running on :8080?"
      );
    } finally {
      setLoading(false);
    }
  };

  const isFake = result?.prediction?.toLowerCase().includes("fake");

  return (
    <main className="min-h-screen bg-void pb-32">
      <AppHeader />

      <div className="mx-auto max-w-5xl px-6 py-16">
        <p className="font-mono text-xs uppercase tracking-[0.2em] text-accent-blue">
          Detection Engine
        </p>
        <h1 className="mt-3 font-display text-4xl font-medium tracking-tight text-ink sm:text-5xl">
          Upload media to analyze
        </h1>
        <p className="mt-4 max-w-xl text-ink-dim">
          DeepTrace runs your image through face detection, texture and
          frequency analysis, and forgery classification.
        </p>

        <div className="mt-12 grid gap-8 lg:grid-cols-2">
          <div>
            <div
              onDragOver={(e) => {
                e.preventDefault();
                setDragActive(true);
              }}
              onDragLeave={() => setDragActive(false)}
              onDrop={onDrop}
              onClick={() => inputRef.current?.click()}
              className={clsx(
                "glass relative flex aspect-square cursor-pointer flex-col items-center justify-center overflow-hidden rounded-2xl border-2 border-dashed p-8 text-center transition-colors",
                dragActive ? "border-accent-blue" : "border-white/15"
              )}
            >
              {previewUrl ? (
                // eslint-disable-next-line @next/next/no-img-element
                <img
                  src={previewUrl}
                  alt="Selected upload"
                  onLoad={() => setMediaReady(true)}
                  className={clsx(
                    "max-h-full max-w-full rounded-lg object-contain transition-opacity",
                    mediaReady ? "opacity-100" : "opacity-0"
                  )}
                />
              ) : (
                <>
                  <p className="text-sm text-ink-dim">
                    Drag &amp; drop an image, or click to browse
                  </p>
                  <p className="mt-1 font-mono text-xs text-ink-faint">
                    JPG · PNG — max 50MB
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
                    Uploading…
                  </p>
                </div>
              )}

              <input
                ref={inputRef}
                type="file"
                accept="image/*"
                className="hidden"
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) pickFile(f);
                }}
              />
            </div>

            <MagneticButton
              className="mt-6 w-full justify-center"
              onClick={onSubmit}
            >
              {loading ? (
                <span className="inline-flex items-center gap-2">
                  Analyzing
                  <LoadingDots size="sm" />
                </span>
              ) : (
                "Run Detection"
              )}
            </MagneticButton>

            {error && (
              <p className="mt-4 rounded-lg border border-accent-orange/30 bg-accent-orange/10 px-3 py-2 text-sm text-accent-orange">
                {error}
              </p>
            )}
          </div>

          <div className="glass rounded-2xl p-6">
            <p className="font-mono text-xs uppercase tracking-widest text-ink-dim">
              Result
            </p>

            {loading && (
              <div className="flex flex-col items-center justify-center gap-3 py-10 text-center">
                <LoadingDots size="lg" className="text-accent-blue" />
                <p className="text-sm text-ink-dim">Running detection…</p>
              </div>
            )}

            {!loading && !result && (
              <p className="mt-6 text-sm text-ink-faint">
                Run detection to see the verdict, confidence score, and
                per-face breakdown here.
              </p>
            )}

            {!loading && result && (
              <div className="mt-4 space-y-6">
                <div className="flex items-center justify-between">
                  <span
                    className={clsx(
                      "rounded-full px-4 py-1.5 text-sm font-medium uppercase tracking-wide",
                      isFake
                        ? "bg-accent-orange/15 text-accent-orange"
                        : "bg-accent-blue/15 text-accent-blue"
                    )}
                  >
                    {result.prediction}
                  </span>
                  <span className="font-mono text-2xl text-ink">
                    {Math.round(result.confidence)}%
                  </span>
                </div>

                {result.processed_image_url && (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img
                    src={mediaUrl(result.processed_image_url) ?? undefined}
                    alt="Processed detection result"
                    className="w-full rounded-lg"
                  />
                )}

                <div>
                  <p className="font-mono text-xs uppercase tracking-widest text-ink-dim">
                    Faces detected: {result.face_count}
                  </p>
                  <div className="mt-3 space-y-2">
                    {result.faces?.map((face, i) => (
                      <div
                        key={i}
                        className="flex items-center justify-between rounded-lg border border-white/10 px-3 py-2 text-sm"
                      >
                        <span className="text-ink-dim">Face {i + 1}</span>
                        <span className="text-ink">{face.label}</span>
                        <span className="font-mono text-accent-orange">
                          {face.fake_confidence}%
                        </span>
                      </div>
                    ))}
                  </div>
                </div>

                {result.timing_ms && (
                  <p className="font-mono text-xs text-ink-faint">
                    Detect {result.timing_ms.detect}ms · Classify{" "}
                    {result.timing_ms.classify}ms · Total {result.timing_ms.total}ms
                  </p>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </main>
  );
}

export default function DetectPage() {
  return (
    <RequireAuth>
      <DetectPageContent />
    </RequireAuth>
  );
}
