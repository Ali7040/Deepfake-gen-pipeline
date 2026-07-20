"use client";

import dynamic from "next/dynamic";
import RevealText from "@/components/ui/RevealText";

const NeuralNetworkScene = dynamic(
  () => import("@/components/three/NeuralNetworkScene"),
  { ssr: false }
);

export default function NeuralNetwork() {
  return (
    <section className="relative h-[120vh] overflow-hidden bg-void">
      <div className="absolute inset-0">
        <NeuralNetworkScene className="h-full w-full" />
      </div>
      <div className="pointer-events-none absolute inset-0 bg-gradient-to-b from-void via-transparent to-void" />
      <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center px-6 text-center">
        <p className="font-mono text-xs uppercase tracking-[0.2em] text-accent-blue">
          Neural Architecture
        </p>
        <RevealText
          as="h2"
          className="mx-auto mt-4 max-w-2xl text-balance font-display text-4xl font-medium leading-[1.05] tracking-tight text-ink sm:text-5xl"
        >
          Millions of parameters. One question.
        </RevealText>
        <p className="mx-auto mt-6 max-w-lg text-ink-dim">
          The network constantly reorganizes itself — every connection a
          learned signal separating real light from synthesized light.
        </p>
      </div>
    </section>
  );
}
