"use client";

import { useEffect, useRef } from "react";
import Link from "next/link";
import { gsap } from "@/lib/gsap";
import { useAuth } from "@/lib/auth-context";

export default function Navbar() {
  const navRef = useRef<HTMLElement>(null);
  const { user, loading } = useAuth();

  useEffect(() => {
    let lastY = window.scrollY;
    const onScroll = () => {
      const y = window.scrollY;
      const goingDown = y > lastY && y > 120;
      gsap.to(navRef.current, {
        yPercent: goingDown ? -100 : 0,
        duration: 0.4,
        ease: "power2.out",
      });
      lastY = y;
    };
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  return (
    <nav
      ref={navRef}
      className="fixed inset-x-0 top-0 z-50 flex items-center justify-between px-6 py-5 sm:px-10"
    >
      <span className="font-display text-sm font-medium tracking-[0.2em] text-ink">
        DEEPTRACE
      </span>
      <div className="flex items-center gap-6 sm:gap-8">
        <div className="hidden items-center gap-8 font-mono text-xs uppercase tracking-widest text-ink-dim sm:flex">
          <a data-cursor-hover href="#detection-engine" className="hover:text-accent-blue">
            Detection
          </a>
          <a data-cursor-hover href="#generate" className="hover:text-accent-blue">
            Generation
          </a>
          <a data-cursor-hover href="#stats" className="hover:text-accent-blue">
            Research
          </a>
        </div>

        {!loading && (
          <div className="flex items-center gap-3 font-mono text-xs uppercase tracking-widest">
            {user ? (
              <Link
                href="/detect"
                data-cursor-hover
                className="rounded-full border border-white/20 px-4 py-2 text-ink transition-colors hover:border-accent-blue hover:text-accent-blue"
              >
                Dashboard
              </Link>
            ) : (
              <>
                <Link
                  href="/login"
                  data-cursor-hover
                  className="text-ink-dim transition-colors hover:text-ink"
                >
                  Log in
                </Link>
                <Link
                  href="/signup"
                  data-cursor-hover
                  className="rounded-full bg-ink px-4 py-2 text-void transition-colors hover:bg-accent-blue"
                >
                  Sign up
                </Link>
              </>
            )}
          </div>
        )}
      </div>
    </nav>
  );
}
