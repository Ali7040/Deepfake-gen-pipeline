"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import clsx from "clsx";
import { useAuth } from "@/lib/auth-context";

export default function AppHeader() {
  const { user, logout } = useAuth();
  const pathname = usePathname();
  const router = useRouter();

  const linkClass = (href: string) =>
    clsx(
      "font-mono text-xs uppercase tracking-widest transition-colors",
      pathname === href ? "text-accent-blue" : "text-ink-dim hover:text-ink"
    );

  return (
    <header className="sticky top-0 z-50 border-b border-white/10 bg-void/80 backdrop-blur">
      <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-4">
        <Link
          href="/"
          className="font-display text-sm font-medium tracking-[0.2em] text-ink"
        >
          DEEPTRACE
        </Link>

        <nav className="flex items-center gap-6">
          <Link href="/detect" className={linkClass("/detect")}>
            Detect
          </Link>
          <Link href="/generate" className={linkClass("/generate")}>
            Generate
          </Link>
          {user && (
            <div className="flex items-center gap-4 border-l border-white/10 pl-6">
              <span className="hidden font-mono text-xs text-ink-dim sm:inline">
                {user.name || user.email}
              </span>
              <button
                onClick={() => {
                  logout();
                  router.push("/");
                }}
                data-cursor-hover
                className="font-mono text-xs uppercase tracking-widest text-ink-dim transition-colors hover:text-accent-orange"
              >
                Log out
              </button>
            </div>
          )}
        </nav>
      </div>
    </header>
  );
}
