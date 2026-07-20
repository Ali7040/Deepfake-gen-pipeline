"use client";

import { useEffect, useRef } from "react";
import { ensureGsap, gsap, ScrollTrigger } from "@/lib/gsap";
import RevealText from "@/components/ui/RevealText";
import { stats } from "@/lib/content";

export default function Stats() {
  const sectionRef = useRef<HTMLDivElement>(null);
  const valueRefs = useRef<Array<HTMLSpanElement | null>>([]);
  const ringRefs = useRef<Array<SVGCircleElement | null>>([]);

  useEffect(() => {
    ensureGsap();
    const circumference = 2 * Math.PI * 42;
    const ctx = gsap.context(() => {
      stats.forEach((stat, i) => {
        const state = { value: 0 };
        const pct = Math.min(stat.value, 100) / 100;

        gsap.set(ringRefs.current[i], {
          strokeDasharray: circumference,
          strokeDashoffset: circumference,
        });

        gsap.to(state, {
          value: stat.value,
          duration: 1.8,
          ease: "power3.out",
          scrollTrigger: {
            trigger: sectionRef.current,
            start: "top 60%",
            toggleActions: "play none none reverse",
          },
          onUpdate: () => {
            const el = valueRefs.current[i];
            if (el) {
              const display =
                stat.value % 1 === 0
                  ? Math.round(state.value)
                  : state.value.toFixed(1);
              el.textContent = `${display}${stat.suffix}`;
            }
          },
        });

        gsap.to(ringRefs.current[i], {
          strokeDashoffset: circumference * (1 - pct),
          duration: 1.8,
          ease: "power3.out",
          scrollTrigger: {
            trigger: sectionRef.current,
            start: "top 60%",
            toggleActions: "play none none reverse",
          },
        });
      });
    }, sectionRef);
    return () => ctx.revert();
  }, []);

  return (
    <section ref={sectionRef} className="relative bg-void px-6 py-32 sm:py-40">
      <div className="mx-auto max-w-6xl">
        <p className="text-center font-mono text-xs uppercase tracking-[0.2em] text-accent-blue">
          Benchmarked
        </p>
        <RevealText
          as="h2"
          className="mx-auto mt-4 max-w-2xl text-balance text-center font-display text-4xl font-medium leading-[1.05] tracking-tight text-ink sm:text-5xl"
        >
          Numbers that hold up.
        </RevealText>

        <div className="mt-20 grid grid-cols-2 gap-x-6 gap-y-14 sm:grid-cols-3 lg:grid-cols-6">
          {stats.map((stat, i) => (
            <div key={stat.id} className="flex flex-col items-center text-center">
              <div className="relative h-24 w-24">
                <svg viewBox="0 0 100 100" className="h-full w-full -rotate-90">
                  <circle
                    cx="50"
                    cy="50"
                    r="42"
                    fill="none"
                    stroke="rgba(255,255,255,0.08)"
                    strokeWidth="4"
                  />
                  <circle
                    ref={(el) => {
                      ringRefs.current[i] = el;
                    }}
                    cx="50"
                    cy="50"
                    r="42"
                    fill="none"
                    stroke="#4ab3ff"
                    strokeWidth="4"
                    strokeLinecap="round"
                  />
                </svg>
                <span
                  ref={(el) => {
                    valueRefs.current[i] = el;
                  }}
                  className="absolute inset-0 flex items-center justify-center font-mono text-sm text-ink"
                >
                  0{stat.suffix}
                </span>
              </div>
              <p className="mt-4 font-mono text-[11px] uppercase tracking-wider text-ink-dim">
                {stat.label}
              </p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
