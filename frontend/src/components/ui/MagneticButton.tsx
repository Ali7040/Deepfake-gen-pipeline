"use client";

import { useRef } from "react";
import { gsap } from "@/lib/gsap";
import clsx from "clsx";

export default function MagneticButton({
  children,
  className,
  variant = "solid",
  onClick,
  href,
}: {
  children: React.ReactNode;
  className?: string;
  variant?: "solid" | "outline";
  onClick?: () => void;
  href?: string;
}) {
  const ref = useRef<HTMLAnchorElement | HTMLButtonElement | null>(null);

  const onMove = (e: React.PointerEvent) => {
    const el = ref.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const x = e.clientX - rect.left - rect.width / 2;
    const y = e.clientY - rect.top - rect.height / 2;
    gsap.to(el, { x: x * 0.35, y: y * 0.45, duration: 0.5, ease: "power3.out" });
  };

  const onLeave = () => {
    const el = ref.current;
    if (!el) return;
    gsap.to(el, { x: 0, y: 0, duration: 0.6, ease: "elastic.out(1, 0.4)" });
  };

  const classes = clsx(
    "relative inline-flex items-center justify-center px-8 py-4 rounded-full text-sm tracking-wide font-medium transition-colors duration-300 cursor-pointer",
    variant === "solid"
      ? "bg-ink text-void hover:bg-accent-blue"
      : "border border-white/20 text-ink hover:border-accent-blue hover:text-accent-blue",
    className
  );

  if (href) {
    return (
      <a
        ref={ref as React.RefObject<HTMLAnchorElement>}
        href={href}
        data-cursor-hover
        onPointerMove={onMove}
        onPointerLeave={onLeave}
        className={classes}
      >
        {children}
      </a>
    );
  }

  return (
    <button
      ref={ref as React.RefObject<HTMLButtonElement>}
      onClick={onClick}
      data-cursor-hover
      onPointerMove={onMove}
      onPointerLeave={onLeave}
      className={classes}
    >
      {children}
    </button>
  );
}
