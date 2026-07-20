"use client";

import { useEffect, useRef } from "react";
import { ensureGsap, gsap } from "@/lib/gsap";
import FaceScene from "@/components/three/FaceScene";
import MagneticButton from "@/components/ui/MagneticButton";

export default function Hero() {
  const wrapperRef = useRef<HTMLDivElement>(null);
  const heroLayerRef = useRef<HTMLDivElement>(null);
  const line1Ref = useRef<HTMLSpanElement>(null);
  const line2Ref = useRef<HTMLSpanElement>(null);
  const subRef = useRef<HTMLParagraphElement>(null);
  const ctaRef = useRef<HTMLDivElement>(null);

  const genWrapRef = useRef<HTMLDivElement>(null);
  const genLine1Ref = useRef<HTMLHeadingElement>(null);
  const genLine2Ref = useRef<HTMLParagraphElement>(null);

  const progressRef = useRef(0);

  useEffect(() => {
    ensureGsap();
    const ctx = gsap.context(() => {
      gsap.set([line1Ref.current, line2Ref.current], { yPercent: 110 });
      gsap.set(subRef.current, { opacity: 0, y: 16 });
      gsap.set(ctaRef.current, { opacity: 0, y: 16 });

      gsap
        .timeline({ delay: 0.3 })
        .to([line1Ref.current, line2Ref.current], {
          yPercent: 0,
          duration: 1.1,
          stagger: 0.12,
          ease: "power4.out",
        })
        .to(subRef.current, { opacity: 1, y: 0, duration: 0.8, ease: "power3.out" }, "-=0.5")
        .to(ctaRef.current, { opacity: 1, y: 0, duration: 0.8, ease: "power3.out" }, "-=0.55");

      const tl = gsap.timeline({
        scrollTrigger: {
          trigger: wrapperRef.current,
          start: "top top",
          end: "bottom bottom",
          scrub: 1,
        },
      });

      tl.to(heroLayerRef.current, { opacity: 0, y: -40, filter: "blur(6px)", duration: 6 }, 8);

      tl.to(progressRef, { current: 1, duration: 26, ease: "none" }, 16);

      tl.fromTo(
        genLine1Ref.current,
        { opacity: 0, y: 30 },
        { opacity: 1, y: 0, duration: 6, ease: "power3.out" },
        56
      );
      tl.fromTo(
        genLine2Ref.current,
        { opacity: 0, y: 20 },
        { opacity: 1, y: 0, duration: 6, ease: "power3.out" },
        68
      );

      tl.to(genWrapRef.current, { opacity: 0, duration: 4 }, 92);
    }, wrapperRef);

    return () => ctx.revert();
  }, []);

  return (
    <div ref={wrapperRef} className="relative" style={{ height: "300vh" }}>
      <div className="sticky top-0 h-screen w-full overflow-hidden bg-void">
        <FaceScene
          texAUrl="/assets/faces/hero-subject.jpg"
          texBUrl="/assets/faces/identity-b.jpg"
          progressRef={progressRef}
          className="absolute inset-0 h-full w-full"
        />

        <div className="pointer-events-none absolute inset-0 bg-grid opacity-[0.04]" />
        <div className="pointer-events-none absolute inset-0 bg-gradient-to-b from-void via-transparent to-void/90" />
        <div className="pointer-events-none absolute inset-0 bg-gradient-to-r from-void/50 via-transparent to-void/50" />

        <div
          ref={heroLayerRef}
          className="absolute inset-0 z-10 flex flex-col items-center justify-center px-6 text-center"
        >
          <h1 className="font-display text-[12vw] font-medium leading-[0.95] tracking-tight text-ink sm:text-[7rem]">
            <span className="split-line">
              <span ref={line1Ref} className="inline-block">
                CAN YOU TRUST
              </span>
            </span>
            <span className="split-line">
              <span ref={line2Ref} className="inline-block">
                WHAT YOU SEE?
              </span>
            </span>
          </h1>
          <p ref={subRef} className="mt-6 max-w-xl text-balance text-sm text-ink-dim sm:text-base">
            AI-Powered Deepfake Detection &amp; Generation Platform
          </p>
          <div ref={ctaRef} className="mt-10 flex flex-wrap items-center justify-center gap-4">
            <MagneticButton href="/detect">Explore Detection</MagneticButton>
            <MagneticButton href="/generate" variant="outline">
              Generate Deepfake
            </MagneticButton>
          </div>
        </div>

        <div
          ref={genWrapRef}
          className="pointer-events-none absolute inset-0 z-10 flex flex-col items-center justify-end pb-24 text-center"
        >
          <h2
            ref={genLine1Ref}
            className="font-display text-[9vw] font-medium leading-none tracking-tight text-ink opacity-0 sm:text-7xl"
          >
            ONE FACE. TWO IDENTITIES.
          </h2>
          <p ref={genLine2Ref} className="mt-4 text-lg text-accent-orange opacity-0 sm:text-xl">
            Reality can be generated.
          </p>
        </div>
      </div>
    </div>
  );
}
