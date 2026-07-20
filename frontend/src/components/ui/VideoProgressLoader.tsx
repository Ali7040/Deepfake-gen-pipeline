import type { SwapProgress } from "@/lib/api";
import LoadingDots from "./LoadingDots";

export default function VideoProgressLoader({ progress }: { progress: SwapProgress | null }) {
  const total = progress?.total ?? 0;
  const done = progress?.done ?? 0;
  const pct = total > 0 ? Math.min(100, Math.round((done / total) * 100)) : 0;
  const status = progress?.status ?? "queued";

  return (
    <div className="flex flex-col items-center justify-center gap-4 py-10 text-center">
      <LoadingDots size="lg" className="text-accent-blue" />

      <div className="w-full max-w-xs">
        <div className="flex items-center justify-between">
          <p className="font-mono text-xs uppercase tracking-widest text-accent-blue">
            {status}
          </p>
          <p className="font-mono text-xs text-ink-dim">{pct}%</p>
        </div>
        <div className="mt-2 h-1.5 w-full overflow-hidden rounded-full bg-white/10">
          <div
            className="h-full rounded-full bg-gradient-to-r from-accent-blue to-accent-orange transition-all duration-300"
            style={{ width: `${pct}%` }}
          />
        </div>
        <p className="mt-2 text-xs text-ink-faint">
          {total > 0 ? `Frame ${done} of ${total}` : "Starting up…"}
          {progress?.eta_seconds != null && progress.eta_seconds > 0
            ? ` — ~${progress.eta_seconds}s left`
            : ""}
          {progress?.fps_proc ? ` · ${progress.fps_proc.toFixed(1)} fps` : ""}
        </p>
      </div>
    </div>
  );
}
