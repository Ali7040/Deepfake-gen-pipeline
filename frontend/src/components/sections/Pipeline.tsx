"use client";

import { useEffect, useRef } from "react";
import { ensureGsap, gsap, ScrollTrigger } from "@/lib/gsap";
import RevealText from "@/components/ui/RevealText";
import { pipelineStages } from "@/lib/content";

export default function Pipeline() {
  const sectionRef = useRef<HTMLDivElement>(null);
  const lineRef = useRef<HTMLDivElement>(null);
  const dotRefs = useRef<Array<HTMLDivElement | null>>([]);

  useEffect(() => {
    ensureGsap();
    const ctx = gsap.context(() => {
      gsap.fromTo(
        lineRef.current,
        { scaleX: 0 },
        {
          scaleX: 1,
          duration: 1.6,
          ease: "power2.inOut",
          transformOrigin: "left center",
          scrollTrigger: {
            trigger: sectionRef.current,
            start: "top 60%",
            toggleActions: "play none none reverse",
          },
        }
      );

      gsap.fromTo(
        dotRefs.current,
        { opacity: 0, y: 16 },
        {
          opacity: 1,
          y: 0,
          duration: 0.6,
          stagger: 0.12,
          ease: "power2.out",
          scrollTrigger: {
            trigger: sectionRef.current,
            start: "top 55%",
            toggleActions: "play none none reverse",
          },
        }
      );
    }, sectionRef);
    return () => ctx.revert();
  }, []);

  return (
    <section ref={sectionRef} className="relative bg-void px-6 py-32 sm:py-40">
      <div className="mx-auto max-w-6xl">
        <p className="text-center font-mono text-xs uppercase tracking-[0.2em] text-accent-blue">
          Inference Pipeline
        </p>
        <RevealText
          as="h2"
          className="mx-auto mt-4 max-w-2xl text-balance text-center font-display text-4xl font-medium leading-[1.05] tracking-tight text-ink sm:text-5xl"
        >
          From pixels to prediction.
        </RevealText>

        <div className="relative mt-24 hidden lg:block">
          <div className="absolute left-0 right-0 top-5 h-px bg-white/10" />
          <div
            ref={lineRef}
            className="absolute left-0 right-0 top-5 h-px scale-x-0 bg-gradient-to-r from-accent-blue to-accent-orange"
          />
          <div className="relative grid grid-cols-7 gap-4">
            {pipelineStages.map((stage, i) => (
              <div
                key={stage.id}
                ref={(el) => {
                  dotRefs.current[i] = el;
                }}
                className="flex flex-col items-center text-center opacity-0"
              >
                <span className="relative z-10 h-2.5 w-2.5 rounded-full bg-accent-blue shadow-[0_0_10px_2px_rgba(74,179,255,0.6)]" />
                <p className="mt-6 font-mono text-[11px] uppercase leading-tight tracking-wide text-ink-dim">
                  {stage.label}
                </p>
              </div>
            ))}
          </div>
        </div>

        <div className="mt-16 flex flex-col gap-6 lg:hidden">
          {pipelineStages.map((stage, i) => (
            <div
              key={stage.id}
              ref={(el) => {
                dotRefs.current[i] = el;
              }}
              className="flex items-center gap-4 opacity-0"
            >
              <span className="h-2.5 w-2.5 shrink-0 rounded-full bg-accent-blue shadow-[0_0_10px_2px_rgba(74,179,255,0.6)]" />
              <p className="font-mono text-xs uppercase tracking-wide text-ink-dim">
                {stage.label}
              </p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
