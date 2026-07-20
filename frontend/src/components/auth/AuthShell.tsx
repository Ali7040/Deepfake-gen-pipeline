import Link from "next/link";
import Image from "next/image";

export default function AuthShell({
  eyebrow,
  title,
  subtitle,
  children,
  footer,
}: {
  eyebrow: string;
  title: string;
  subtitle: string;
  children: React.ReactNode;
  footer: React.ReactNode;
}) {
  return (
    <main className="relative flex min-h-screen items-center justify-center overflow-hidden bg-void px-6 py-20">
      <div className="absolute inset-0">
        <Image
          src="/assets/faces/hero-subject.jpg"
          alt=""
          fill
          priority
          sizes="100vw"
          className="kenburns object-cover object-top opacity-30 grayscale"
        />
      </div>
      <div className="bg-grid pointer-events-none absolute inset-0 opacity-40" />
      <div className="pointer-events-none absolute inset-0 bg-gradient-to-b from-void via-void/70 to-void" />
      <div className="pointer-events-none absolute inset-0 bg-gradient-to-r from-void via-transparent to-void" />
      <div
        className="pointer-events-none absolute left-1/2 top-1/3 h-[420px] w-[420px] -translate-x-1/2 -translate-y-1/2 rounded-full opacity-40"
        style={{
          background:
            "radial-gradient(circle, rgba(74,179,255,0.25) 0%, transparent 70%)",
          filter: "blur(40px)",
        }}
      />
      <div className="grain-overlay" aria-hidden="true" />

      <div className="relative w-full max-w-md">
        <Link
          href="/"
          className="mb-8 inline-block font-display text-sm font-medium tracking-[0.2em] text-ink"
        >
          DEEPTRACE
        </Link>

        <div className="glass rounded-2xl p-8 sm:p-10">
          <p className="font-mono text-xs uppercase tracking-[0.2em] text-accent-blue">
            {eyebrow}
          </p>
          <h1 className="mt-3 font-display text-3xl font-medium tracking-tight text-ink">
            {title}
          </h1>
          <p className="mt-2 text-sm text-ink-dim">{subtitle}</p>

          <div className="mt-8">{children}</div>
        </div>

        <p className="mt-6 text-center text-sm text-ink-dim">{footer}</p>
      </div>
    </main>
  );
}
