import Navbar from "@/components/layout/Navbar";
import Hero from "@/components/sections/Hero";
import DetectionEngine from "@/components/sections/DetectionEngine";
import BeforeAfterSlider from "@/components/sections/BeforeAfterSlider";
import NeuralNetwork from "@/components/sections/NeuralNetwork";
import Pipeline from "@/components/sections/Pipeline";
import Stats from "@/components/sections/Stats";
import FinalSection from "@/components/sections/FinalSection";

export default function Home() {
  return (
    <>
      <Navbar />
      <main>
        <Hero />
        <DetectionEngine />
        <div id="generate">
          <BeforeAfterSlider />
        </div>
        <NeuralNetwork />
        <Pipeline />
        <div id="stats">
          <Stats />
        </div>
        <FinalSection />
      </main>
    </>
  );
}
