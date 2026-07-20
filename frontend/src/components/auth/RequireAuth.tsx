"use client";

import { useEffect } from "react";
import { useRouter, usePathname } from "next/navigation";
import { useAuth } from "@/lib/auth-context";
import LoadingDots from "@/components/ui/LoadingDots";

export default function RequireAuth({ children }: { children: React.ReactNode }) {
  const { user, loading } = useAuth();
  const router = useRouter();
  const pathname = usePathname();

  useEffect(() => {
    if (!loading && !user) {
      router.replace(`/login?next=${encodeURIComponent(pathname)}`);
    }
  }, [loading, user, router, pathname]);

  if (loading || !user) {
    return (
      <main className="flex min-h-screen flex-col items-center justify-center gap-3 bg-void">
        <LoadingDots size="lg" className="text-accent-blue" />
        <p className="font-mono text-xs uppercase tracking-widest text-ink-dim">
          Checking session…
        </p>
      </main>
    );
  }

  return <>{children}</>;
}
