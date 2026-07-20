"use client";

import { useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import AuthShell from "@/components/auth/AuthShell";
import MagneticButton from "@/components/ui/MagneticButton";
import LoadingDots from "@/components/ui/LoadingDots";
import { useAuth } from "@/lib/auth-context";
import { friendlyErrorMessage } from "@/lib/api";

export default function SignupPage() {
  const { register } = useAuth();
  const router = useRouter();
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  const onSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setSubmitting(true);
    try {
      await register(email, password, name);
      router.push("/detect");
    } catch (err) {
      setError(friendlyErrorMessage(err));
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <AuthShell
      eyebrow="Get started"
      title="Create an account"
      subtitle="Run deepfake detection and generation on your own media."
      footer={
        <>
          Already have an account?{" "}
          <Link href="/login" className="text-accent-blue hover:underline">
            Log in
          </Link>
        </>
      }
    >
      <form onSubmit={onSubmit} className="space-y-4">
        <div>
          <label className="mb-1.5 block font-mono text-xs uppercase tracking-wider text-ink-dim">
            Name
          </label>
          <input
            type="text"
            required
            value={name}
            onChange={(e) => setName(e.target.value)}
            className="w-full rounded-lg border border-white/10 bg-white/5 px-4 py-3 text-sm text-ink outline-none transition-colors focus:border-accent-blue"
            placeholder="Ada Lovelace"
          />
        </div>
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
            minLength={6}
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="w-full rounded-lg border border-white/10 bg-white/5 px-4 py-3 text-sm text-ink outline-none transition-colors focus:border-accent-blue"
            placeholder="At least 6 characters"
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
              Creating account
              <LoadingDots size="sm" />
            </span>
          ) : (
            "Create account"
          )}
        </MagneticButton>
      </form>
    </AuthShell>
  );
}
