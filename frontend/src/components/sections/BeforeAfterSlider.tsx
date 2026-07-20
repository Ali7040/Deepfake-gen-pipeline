"use client";

import { useRef, useState, useCallback } from "react";
import Image from "next/image";
import RevealText from "@/components/ui/RevealText";

export default function BeforeAfterSlider() {
  const [percent, setPercent] = useState(50);
  const containerRef = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);

  const updateFromClientX = useCallback((clientX: number) => {
    const el = containerRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const pct = ((clientX - rect.left) / rect.width) * 100;
    setPercent(Math.min(100, Math.max(0, pct)));
  }, []);

  const onPointerDown = (e: React.PointerEvent) => {
    dragging.current = true;
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
    updateFromClientX(e.clientX);
  };
  const onPointerMove = (e: React.PointerEvent) => {
    if (!dragging.current) return;
    updateFromClientX(e.clientX);
  };
  const onPointerUp = () => {
    dragging.current = false;
  };

  return (
    <section className="bg-grid relative overflow-hidden bg-void px-6 py-32 sm:py-40">
      <div className="pointer-events-none absolute inset-0 bg-gradient-to-b from-void via-transparent to-void" />
      <div
        className="pointer-events-none absolute left-1/2 top-1/2 h-[520px] w-[520px] -translate-x-1/2 -translate-y-1/2 rounded-full opacity-30"
        style={{
          background:
            "radial-gradient(circle, rgba(74,179,255,0.22) 0%, transparent 70%)",
          filter: "blur(50px)",
        }}
      />

      <div className="relative mx-auto max-w-5xl text-center">
        <p className="font-mono text-xs uppercase tracking-[0.2em] text-accent-blue">
          Original vs. Generated
        </p>
        <RevealText
          as="h2"
          className="mx-auto mt-4 max-w-2xl font-display text-4xl font-medium leading-[1.05] tracking-tight text-ink sm:text-5xl"
        >
          Drag the line. Decide for yourself.
        </RevealText>

        <div
          ref={containerRef}
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          className="relative mx-auto mt-12 aspect-[4/3] w-full max-w-2xl cursor-ew-resize select-none overflow-hidden rounded-2xl"
        >
          <Image
            src="/assets/faces/swap-result-generated.webp"
            alt="Generated face"
            fill
            sizes="(max-width: 768px) 100vw, 720px"
            className="object-cover"
            draggable={false}
          />

          <div
            className="absolute inset-0 overflow-hidden"
            style={{ clipPath: `inset(0 ${100 - percent}% 0 0)` }}
          >
            <Image
              src="/assets/faces/swap-tomcruise-original.webp"
              alt="Original face"
              fill
              sizes="(max-width: 768px) 100vw, 720px"
              className="object-cover"
              draggable={false}
            />
          </div>

          <div
            className="absolute top-0 h-full w-px bg-accent-blue"
            style={{ left: `${percent}%` }}
          >
            <div className="absolute top-1/2 left-1/2 flex h-10 w-10 -translate-x-1/2 -translate-y-1/2 items-center justify-center rounded-full border border-accent-blue bg-void/80 backdrop-blur">
              <span className="text-xs text-accent-blue">↔</span>
            </div>
          </div>

          <span className="glass absolute left-3 top-3 rounded px-2 py-1 font-mono text-[10px] uppercase tracking-wider text-ink-dim">
            Original
          </span>
          <span className="glass absolute right-3 top-3 rounded px-2 py-1 font-mono text-[10px] uppercase tracking-wider text-ink-dim">
            Generated
          </span>
        </div>
      </div>
    </section>
  );
}
