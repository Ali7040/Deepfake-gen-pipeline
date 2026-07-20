"use client";

import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import { SplitText } from "gsap/SplitText";

let registered = false;

export function ensureGsap() {
  if (registered) return gsap;
  gsap.registerPlugin(ScrollTrigger, SplitText);
  registered = true;
  return gsap;
}

export { gsap, ScrollTrigger, SplitText };
