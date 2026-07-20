import type { Metadata } from "next";
import { Space_Grotesk, Inter, JetBrains_Mono } from "next/font/google";
import "./globals.css";
import SmoothScrollProvider from "@/components/layout/SmoothScrollProvider";
import GrainOverlay from "@/components/layout/GrainOverlay";
import { AuthProvider } from "@/lib/auth-context";

const display = Space_Grotesk({
  variable: "--font-neue-montreal",
  subsets: ["latin"],
  weight: ["300", "400", "500", "700"],
});

const body = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
  weight: ["300", "400", "500", "600"],
});

const mono = JetBrains_Mono({
  variable: "--font-jetbrains-mono",
  subsets: ["latin"],
  weight: ["400", "500"],
});

export const metadata: Metadata = {
  title: "DeepTrace — Can You Trust What You See?",
  description:
    "DeepTrace is an AI-powered platform for deepfake detection and generation, using explainable visual analysis and state-of-the-art neural models.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${display.variable} ${body.variable} ${mono.variable} h-full antialiased`}
    >
      <body
        className="min-h-full flex flex-col bg-void text-ink"
        suppressHydrationWarning
      >
        <AuthProvider>
          <SmoothScrollProvider>
            <GrainOverlay />
            {children}
          </SmoothScrollProvider>
        </AuthProvider>
      </body>
    </html>
  );
}
