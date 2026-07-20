"use client";

import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { useTexture } from "@react-three/drei";
import { useMemo, useRef } from "react";
import * as THREE from "three";
import { vertexShader, fragmentShader } from "@/three/shaders/faceMorph";

function FaceMesh({
  texAUrl,
  texBUrl,
  progressRef,
}: {
  texAUrl: string;
  texBUrl: string;
  progressRef: React.MutableRefObject<number>;
}) {
  const [texA, texB] = useTexture([texAUrl, texBUrl]);
  const { viewport } = useThree();
  const materialRef = useRef<THREE.ShaderMaterial>(null);
  const mouse = useRef(new THREE.Vector2(0, 0));
  const hoverLerp = useRef(0);
  const prevProgress = useRef(0);
  const aberrationLerp = useRef(0);

  texA.colorSpace = THREE.SRGBColorSpace;
  texB.colorSpace = THREE.SRGBColorSpace;

  const uniforms = useMemo(
    () => ({
      uTexA: { value: texA },
      uTexB: { value: texB },
      uProgress: { value: 0 },
      uTime: { value: 0 },
      uMouse: { value: new THREE.Vector2(0, 0) },
      uHover: { value: 0 },
      uAberration: { value: 0 },
      uResolution: { value: new THREE.Vector2(1, 1) },
      uImageSizeA: {
        value: new THREE.Vector2(
          (texA.image as HTMLImageElement)?.width || 1,
          (texA.image as HTMLImageElement)?.height || 1
        ),
      },
      uImageSizeB: {
        value: new THREE.Vector2(
          (texB.image as HTMLImageElement)?.width || 1,
          (texB.image as HTMLImageElement)?.height || 1
        ),
      },
    }),
    [texA, texB]
  );

  useFrame((state, delta) => {
    const mat = materialRef.current;
    if (!mat) return;

    const target = state.pointer;
    mouse.current.lerp(new THREE.Vector2(target.x, target.y), 0.06);
    hoverLerp.current = THREE.MathUtils.lerp(hoverLerp.current, 1, 0.02);

    const progress = progressRef.current;
    const velocity = Math.abs(progress - prevProgress.current) / Math.max(delta, 0.001);
    prevProgress.current = progress;
    aberrationLerp.current = THREE.MathUtils.lerp(
      aberrationLerp.current,
      Math.min(velocity * 8, 1),
      0.15
    );

    mat.uniforms.uProgress.value = progress;
    mat.uniforms.uTime.value = state.clock.elapsedTime;
    mat.uniforms.uMouse.value.copy(mouse.current);
    mat.uniforms.uHover.value = hoverLerp.current;
    mat.uniforms.uAberration.value = aberrationLerp.current;
    mat.uniforms.uResolution.value.set(viewport.width, viewport.height);
  });

  return (
    <mesh scale={[viewport.width, viewport.height, 1]}>
      <planeGeometry args={[1, 1, 1, 1]} />
      <shaderMaterial
        ref={materialRef}
        vertexShader={vertexShader}
        fragmentShader={fragmentShader}
        uniforms={uniforms}
      />
    </mesh>
  );
}

export default function FaceScene({
  texAUrl,
  texBUrl,
  progressRef,
  className,
}: {
  texAUrl: string;
  texBUrl: string;
  progressRef: React.MutableRefObject<number>;
  className?: string;
}) {
  return (
    <Canvas
      className={className}
      orthographic
      camera={{ position: [0, 0, 1], zoom: 1, near: 0.01, far: 10 }}
      dpr={[1, 1.75]}
      gl={{ antialias: true, powerPreference: "high-performance" }}
    >
      <FaceMesh texAUrl={texAUrl} texBUrl={texBUrl} progressRef={progressRef} />
    </Canvas>
  );
}
