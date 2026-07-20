"use client";

import { useEffect, useRef } from "react";
import Image from "next/image";
import { ensureGsap, gsap, ScrollTrigger } from "@/lib/gsap";
import RevealText from "@/components/ui/RevealText";

const heatZones = [
  { top: "22%", left: "38%", size: 90, delay: 0 },
  { top: "30%", left: "58%", size: 70, delay: 0.15 },
  { top: "44%", left: "46%", size: 110, delay: 0.3 },
  { top: "52%", left: "34%", size: 60, delay: 0.45 },
];

export default function DetectionEngine() {
  const sectionRef = useRef<HTMLDivElement>(null);
  const counterRef = useRef<HTMLSpanElement>(null);
  const barRef = useRef<HTMLDivElement>(null);
  const zonesRef = useRef<HTMLDivElement>(null);
  const dropRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    ensureGsap();
    const ctx = gsap.context(() => {
      const counterState = { value: 0 };
      const tl = gsap.timeline({
        scrollTrigger: {
          trigger: sectionRef.current,
          start: "top 55%",
          toggleActions: "play none none reverse",
        },
      });

      tl.fromTo(
        dropRef.current,
        { opacity: 0, y: 20 },
        { opacity: 1, y: 0, duration: 0.6, ease: "power3.out" }
      )
        .to(barRef.current, { scaleX: 1, duration: 1.6, ease: "power3.inOut" }, 0.2)
        .to(
          counterState,
          {
            value: 99,
            duration: 1.6,
            ease: "power3.inOut",
            onUpdate: () => {
              if (counterRef.current) {
                counterRef.current.textContent = `${Math.round(counterState.value)}%`;
              }
            },
          },
          0.2
        )
        .fromTo(
          zonesRef.current?.children ?? [],
          { opacity: 0, scale: 0.6 },
          { opacity: 1, scale: 1, duration: 0.8, stagger: 0.12, ease: "power2.out" },
          0.6
        );
    }, sectionRef);

    return () => ctx.revert();
  }, []);

  return (
    <section
      id="detection-engine"
      ref={sectionRef}
      className="relative bg-void px-6 py-32 sm:py-40"
    >
      <div className="mx-auto grid max-w-6xl gap-16 lg:grid-cols-2 lg:items-center">
        <div>
          <p className="font-mono text-xs uppercase tracking-[0.2em] text-accent-blue">
            Detection Engine
          </p>
          <RevealText
            as="h2"
            className="mt-4 font-display text-4xl font-medium leading-[1.05] tracking-tight text-ink sm:text-5xl"
          >
            Upload. Analyze. Reveal the truth.
          </RevealText>
          <p className="mt-6 max-w-md text-ink-dim">
            Every frame is decomposed into facial landmarks, texture frequency
            maps, and lighting-consistency signals — scored in real time by an
            ensemble of vision transformers.
          </p>

          <div ref={dropRef} className="glass mt-10 rounded-2xl p-6 opacity-0">
            <div className="flex items-center justify-between border border-dashed border-white/15 rounded-xl px-6 py-10 text-center">
              <div className="mx-auto">
                <p className="text-sm text-ink-dim">
                  Drag &amp; drop an image or video
                </p>
                <p className="mt-1 font-mono text-xs text-ink-faint">
                  JPG · PNG · MP4 — max 50MB
                </p>
              </div>
            </div>

            <div className="mt-6 flex items-center justify-between">
              <span className="font-mono text-xs uppercase tracking-widest text-ink-dim">
                Confidence Score
              </span>
              <span ref={counterRef} className="font-mono text-2xl text-accent-orange">
                0%
              </span>
            </div>
            <div className="mt-3 h-1.5 w-full overflow-hidden rounded-full bg-white/10">
              <div
                ref={barRef}
                className="h-full origin-left scale-x-0 rounded-full bg-gradient-to-r from-accent-blue to-accent-orange"
              />
            </div>
          </div>
        </div>

        <div className="relative mx-auto aspect-[4/5] w-full max-w-md overflow-hidden rounded-2xl">
          <Image
            src="/assets/faces/analysis-subject.jpg"
            alt="Face under AI analysis"
            fill
            sizes="(max-width: 768px) 100vw, 480px"
            className="object-cover grayscale contrast-125 brightness-[0.55]"
          />
          <div className="absolute inset-0 bg-gradient-to-t from-void via-void/10 to-transparent" />
          <div className="bg-grid absolute inset-0 opacity-20" />
          <div ref={zonesRef} className="absolute inset-0">
            {heatZones.map((z, i) => (
              <span
                key={i}
                className="absolute rounded-full opacity-0"
                style={{
                  top: z.top,
                  left: z.left,
                  width: z.size,
                  height: z.size,
                  background:
                    "radial-gradient(circle, rgba(255,107,53,0.55) 0%, rgba(255,107,53,0.15) 55%, transparent 75%)",
                  filter: "blur(2px)",
                }}
              />
            ))}
          </div>
          <div className="glass absolute bottom-3 left-3 right-3 rounded-lg px-3 py-2">
            <p className="font-mono text-[11px] text-ink-dim">
              heatmap_overlay.render() <span className="text-accent-blue">// forgery likelihood</span>
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}
