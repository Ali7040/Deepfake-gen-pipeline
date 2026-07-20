"use client";

import { Suspense, useState } from "react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import AuthShell from "@/components/auth/AuthShell";
import MagneticButton from "@/components/ui/MagneticButton";
import LoadingDots from "@/components/ui/LoadingDots";
import { useAuth } from "@/lib/auth-context";
import { friendlyErrorMessage } from "@/lib/api";

function LoginForm() {
  const { login } = useAuth();
  const router = useRouter();
  const params = useSearchParams();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  const onSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setSubmitting(true);
    try {
      await login(email, password);
      router.push(params.get("next") || "/detect");
    } catch (err) {
      setError(friendlyErrorMessage(err));
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <AuthShell
      eyebrow="Welcome back"
      title="Log in"
      subtitle="Access detection and generation with your DeepTrace account."
      footer={
        <>
          Don&apos;t have an account?{" "}
          <Link href="/signup" className="text-accent-blue hover:underline">
            Sign up
          </Link>
        </>
      }
    >
      <form onSubmit={onSubmit} className="space-y-4">
        <div>
          <label className="mb-1.5 block font-mono text-xs uppercase tracking-wider text-ink-dim">
            Email
          </label>
          <input
            type="email"
            required
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className="w-full rounded-lg border border-white/10 bg-white/5 px-4 py-3 text-sm text-ink outline-none transition-colors focus:border-accent-blue"
            placeholder="you@example.com"
          />
        </div>
        <div>
          <label className="mb-1.5 block font-mono text-xs uppercase tracking-wider text-ink-dim">
            Password
          </label>
          <input
            type="password"
            required
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="w-full rounded-lg border border-white/10 bg-white/5 px-4 py-3 text-sm text-ink outline-none transition-colors focus:border-accent-blue"
            placeholder="••••••••"
          />
        </div>

        {error && (
          <p className="rounded-lg border border-accent-orange/30 bg-accent-orange/10 px-3 py-2 text-sm text-accent-orange">
            {error}
          </p>
        )}

        <MagneticButton className="mt-2 w-full justify-center">
          {submitting ? (
            <span className="inline-flex items-center gap-2">
              Logging in
              <LoadingDots size="sm" />
            </span>
          ) : (
            "Log in"
          )}
        </MagneticButton>
      </form>
    </AuthShell>
  );
}

export default function LoginPage() {
  return (
    <Suspense fallback={null}>
      <LoginForm />
    </Suspense>
  );
}
