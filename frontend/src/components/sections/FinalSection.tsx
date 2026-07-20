"use client";

import { useEffect, useMemo, useRef } from "react";
import { ensureGsap, gsap, ScrollTrigger } from "@/lib/gsap";
import RevealText from "@/components/ui/RevealText";
import MagneticButton from "@/components/ui/MagneticButton";

function mulberry32(seed: number) {
  return function () {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export default function FinalSection() {
  const sectionRef = useRef<HTMLDivElement>(null);
  const particleWrapRef = useRef<HTMLDivElement>(null);
  const eyeRef = useRef<HTMLDivElement>(null);

  const particles = useMemo(() => {
    const rand = mulberry32(7);
    return Array.from({ length: 90 }, () => ({
      left: rand() * 100,
      top: rand() * 100,
      size: 1 + rand() * 2.5,
      delay: rand() * 1.5,
    }));
  }, []);

  useEffect(() => {
    ensureGsap();
    const ctx = gsap.context(() => {
      const tl = gsap.timeline({
        scrollTrigger: {
          trigger: sectionRef.current,
          start: "top top",
          end: "+=150%",
          scrub: 1,
          pin: true,
        },
      });

      tl.to(
        Array.from(particleWrapRef.current?.children ?? []),
        {
          opacity: 1,
          scale: 1,
          stagger: { each: 0.01, from: "random" },
          duration: 1,
        },
        0
      ).to(
        eyeRef.current,
        { opacity: 0, filter: "blur(20px)", duration: 1 },
        0.6
      );
    }, sectionRef);
    return () => ctx.revert();
  }, []);

  return (
    <section ref={sectionRef} className="relative h-screen overflow-hidden bg-void">
      <div
        ref={eyeRef}
        className="absolute left-1/2 top-1/2 h-40 w-40 -translate-x-1/2 -translate-y-1/2 rounded-full opacity-100"
        style={{
          background:
            "radial-gradient(circle, rgba(74,179,255,0.9) 0%, rgba(74,179,255,0.15) 55%, transparent 75%)",
          filter: "blur(2px)",
        }}
      />

      <div ref={particleWrapRef} className="absolute inset-0">
        {particles.map((p, i) => (
          <span
            key={i}
            className="absolute scale-0 rounded-full bg-accent-blue opacity-0"
            style={{
              left: `${p.left}%`,
              top: `${p.top}%`,
              width: p.size,
              height: p.size,
              transitionDelay: `${p.delay}s`,
            }}
          />
        ))}
      </div>

      <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center px-6 text-center">
        <RevealText
          as="h2"
          className="max-w-3xl text-balance font-display text-4xl font-medium leading-[1.05] tracking-tight text-ink sm:text-6xl"
        >
          NOT EVERYTHING YOU SEE IS REAL.
        </RevealText>
        <p className="mt-6 max-w-xl text-balance text-ink-dim">
          DeepTrace uses advanced artificial intelligence to generate and
          detect deepfakes with explainable visual analysis, neural feature
          extraction, and state-of-the-art detection models.
        </p>
        <div className="pointer-events-auto mt-10">
          <MagneticButton href="/detect">
            Explore the Future of AI
          </MagneticButton>
        </div>
      </div>
    </section>
  );
}
