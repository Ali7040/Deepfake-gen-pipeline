"use client";

import { useEffect, useRef } from "react";
import { ensureGsap, gsap, ScrollTrigger, SplitText } from "@/lib/gsap";
import clsx from "clsx";

export default function RevealText({
  as: Tag = "div",
  children,
  className,
  type = "lines",
  scrub = false,
  delay = 0,
}: {
  as?: React.ElementType;
  children: string;
  className?: string;
  type?: "lines" | "chars" | "words";
  scrub?: boolean;
  delay?: number;
}) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    ensureGsap();

    const split = SplitText.create(el, {
      type: type === "chars" ? "chars,lines" : type === "words" ? "words,lines" : "lines",
      linesClass: "split-line",
      mask: "lines",
    });

    const targets = type === "chars" ? split.chars : type === "words" ? split.words : split.lines;

    const tween = gsap.from(targets, {
      yPercent: 110,
      opacity: 0,
      duration: 1,
      delay,
      stagger: type === "chars" ? 0.015 : 0.08,
      ease: "power4.out",
      scrollTrigger: {
        trigger: el,
        start: "top 85%",
        toggleActions: "play none none reverse",
      },
    });

    return () => {
      tween.scrollTrigger?.kill();
      tween.kill();
      split.revert();
    };
  }, [type, scrub, delay]);

  const Component = Tag as React.FC<
    React.PropsWithChildren<{
      ref: React.RefObject<HTMLDivElement | null>;
      className?: string;
    }>
  >;

  return (
    <Component ref={ref} className={clsx(className)}>
      {children}
    </Component>
  );
}
