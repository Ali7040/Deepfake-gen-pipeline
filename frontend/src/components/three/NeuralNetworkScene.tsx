"use client";

import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { useMemo, useRef } from "react";
import * as THREE from "three";

// Layered feedforward network: input -> hidden layers -> output, like a
// classic DL model diagram rather than an organic point cloud.
const LAYER_COUNTS = [6, 10, 14, 14, 10, 4];
const LAYER_SPAN_X = 8;
const NODE_SPAN_Y = 3.2;
const PULSES_PER_EDGE_BUDGET = 70;

type LayerNode = { x: number; y: number; z: number; isEndpoint: boolean };

function buildLayers() {
  const nodes: LayerNode[][] = [];
  const startX = -LAYER_SPAN_X / 2;
  const stepX = LAYER_SPAN_X / (LAYER_COUNTS.length - 1);

  LAYER_COUNTS.forEach((count, layerIndex) => {
    const x = startX + stepX * layerIndex;
    const isEndpoint = layerIndex === 0 || layerIndex === LAYER_COUNTS.length - 1;
    const layerNodes: LayerNode[] = [];
    for (let i = 0; i < count; i++) {
      const t = count === 1 ? 0.5 : i / (count - 1);
      const y = (t - 0.5) * NODE_SPAN_Y * (count / Math.max(...LAYER_COUNTS));
      const z = (Math.random() - 0.5) * 0.4;
      layerNodes.push({ x, y, z, isEndpoint });
    }
    nodes.push(layerNodes);
  });

  return nodes;
}

function Network() {
  const nodesGroupRef = useRef<THREE.Group>(null);
  const pulsesRef = useRef<THREE.Points>(null);
  const { viewport } = useThree();

  const layers = useMemo(() => buildLayers(), []);

  const { hiddenPositions, endpointPositions, edges } = useMemo(() => {
    const hidden: number[] = [];
    const endpoint: number[] = [];
    const allEdges: { a: LayerNode; b: LayerNode }[] = [];

    layers.forEach((layer, li) => {
      layer.forEach((n) => {
        (n.isEndpoint ? endpoint : hidden).push(n.x, n.y, n.z);
      });
      if (li < layers.length - 1) {
        const next = layers[li + 1];
        layer.forEach((a) => {
          next.forEach((b) => {
            allEdges.push({ a, b });
          });
        });
      }
    });

    return {
      hiddenPositions: new Float32Array(hidden),
      endpointPositions: new Float32Array(endpoint),
      edges: allEdges,
    };
  }, [layers]);

  const edgeLinePositions = useMemo(() => {
    const arr = new Float32Array(edges.length * 2 * 3);
    edges.forEach((e, i) => {
      arr[i * 6] = e.a.x;
      arr[i * 6 + 1] = e.a.y;
      arr[i * 6 + 2] = e.a.z;
      arr[i * 6 + 3] = e.b.x;
      arr[i * 6 + 4] = e.b.y;
      arr[i * 6 + 5] = e.b.z;
    });
    return arr;
  }, [edges]);

  const pulseCount = Math.min(PULSES_PER_EDGE_BUDGET, edges.length);

  const { pulsePositions, pulseState } = useMemo(() => {
    const positions = new Float32Array(pulseCount * 3);
    const state = Array.from({ length: pulseCount }, () => ({
      edgeIndex: Math.floor(Math.random() * edges.length),
      phase: Math.random(),
      speed: 0.25 + Math.random() * 0.35,
    }));
    return { pulsePositions: positions, pulseState: state };
  }, [pulseCount, edges.length]);

  const targetRotation = useRef(new THREE.Vector2(0, 0));

  useFrame((state, delta) => {
    const t = state.clock.elapsedTime;

    if (nodesGroupRef.current) {
      targetRotation.current.x = THREE.MathUtils.lerp(
        targetRotation.current.x,
        state.pointer.y * 0.25,
        0.04
      );
      targetRotation.current.y = THREE.MathUtils.lerp(
        targetRotation.current.y,
        state.pointer.x * 0.35 + t * 0.03,
        0.04
      );
      nodesGroupRef.current.rotation.x = -targetRotation.current.x;
      nodesGroupRef.current.rotation.y = targetRotation.current.y;
    }

    for (let i = 0; i < pulseCount; i++) {
      const p = pulseState[i];
      p.phase += delta * p.speed;
      if (p.phase >= 1) {
        p.phase = 0;
        p.edgeIndex = Math.floor(Math.random() * edges.length);
      }
      const edge = edges[p.edgeIndex];
      pulsePositions[i * 3] = THREE.MathUtils.lerp(edge.a.x, edge.b.x, p.phase);
      pulsePositions[i * 3 + 1] = THREE.MathUtils.lerp(edge.a.y, edge.b.y, p.phase);
      pulsePositions[i * 3 + 2] = THREE.MathUtils.lerp(edge.a.z, edge.b.z, p.phase);
    }
    if (pulsesRef.current) {
      const attr = pulsesRef.current.geometry.attributes.position as THREE.BufferAttribute;
      attr.array.set(pulsePositions);
      attr.needsUpdate = true;
    }
  });

  const scale = Math.min(1, viewport.width / 10);

  return (
    <group ref={nodesGroupRef} scale={scale}>
      <lineSegments>
        <bufferGeometry>
          <bufferAttribute attach="attributes-position" args={[edgeLinePositions, 3]} />
        </bufferGeometry>
        <lineBasicMaterial color="#4ab3ff" transparent opacity={0.12} />
      </lineSegments>

      <points>
        <bufferGeometry>
          <bufferAttribute attach="attributes-position" args={[hiddenPositions, 3]} />
        </bufferGeometry>
        <pointsMaterial size={0.09} color="#4ab3ff" transparent opacity={0.85} sizeAttenuation />
      </points>

      <points>
        <bufferGeometry>
          <bufferAttribute attach="attributes-position" args={[endpointPositions, 3]} />
        </bufferGeometry>
        <pointsMaterial size={0.13} color="#ff6b35" transparent opacity={0.95} sizeAttenuation />
      </points>

      <points ref={pulsesRef}>
        <bufferGeometry>
          <bufferAttribute attach="attributes-position" args={[pulsePositions, 3]} />
        </bufferGeometry>
        <pointsMaterial size={0.06} color="#ffffff" transparent opacity={0.9} sizeAttenuation />
      </points>
    </group>
  );
}

export default function NeuralNetworkScene({ className }: { className?: string }) {
  return (
    <Canvas
      className={className}
      camera={{ position: [0, 0, 7], fov: 50 }}
      dpr={[1, 1.75]}
      gl={{ antialias: true, powerPreference: "high-performance" }}
    >
      <Network />
    </Canvas>
  );
}
