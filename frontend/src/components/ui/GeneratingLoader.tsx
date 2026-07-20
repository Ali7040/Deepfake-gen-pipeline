"use client";

import { useEffect, useState } from "react";
import LoadingDots from "./LoadingDots";

const STAGES = [
  "Detecting faces…",
  "Extracting embeddings…",
  "Aligning geometry…",
  "Synthesizing new face…",
  "Blending & enhancing…",
  "Finalizing output…",
];

export default function GeneratingLoader({
  label = "Generating",
}: {
  label?: string;
}) {
  const [stage, setStage] = useState(0);

  useEffect(() => {
    const id = setInterval(() => {
      setStage((s) => (s + 1) % STAGES.length);
    }, 1800);
    return () => clearInterval(id);
  }, []);

  return (
    <div className="flex flex-col items-center justify-center gap-4 py-10 text-center">
      <LoadingDots size="lg" className="text-accent-blue" />
      <div>
        <p className="font-mono text-xs uppercase tracking-widest text-accent-blue">
          {label}
        </p>
        <p className="mt-1 text-sm text-ink-dim transition-opacity duration-300">
          {STAGES[stage]}
        </p>
        <p className="mt-1 text-xs text-ink-faint">
          This can take a little while on CPU — hang tight.
        </p>
      </div>
    </div>
  );
}
