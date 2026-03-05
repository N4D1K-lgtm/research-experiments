import { useState, useCallback, useRef } from "react";
import { Canvas } from "@react-three/fiber";
import * as THREE from "three";
import type { RenderNode, FilterState } from "../types/trie";
import type { RadialLayout } from "../utils/radialLayout";
import type { LODState } from "../utils/lod";
import { NodeRenderer } from "./NodeRenderer";
import { GlowRenderer } from "./GlowRenderer";
import { EdgeRenderer } from "./EdgeRenderer";
import { LabelOverlay } from "./LabelOverlay";
import { CameraController } from "./CameraController";
import { Picker } from "./Picker";

interface Props {
  nodeMap: Map<number, RenderNode>;
  edges: { source: number; target: number }[];
  layout: RadialLayout | null;
  lod: LODState;
  filter: FilterState;
  onDistanceChange: (d: number) => void;
  onHover: (node: RenderNode | null, x: number, y: number) => void;
  hoveredNodeId: number | null;
  onClick?: (node: RenderNode | null) => void;
}

export function Viewport({
  nodeMap,
  edges,
  layout,
  lod,
  filter,
  onDistanceChange,
  onHover,
  hoveredNodeId,
  onClick,
}: Props) {
  const nodeMeshRef = useRef<THREE.InstancedMesh>(null);
  const [instanceToNode, setInstanceToNode] = useState<number[]>([]);

  const handleInstanceMapUpdate = useCallback((map: number[]) => {
    setInstanceToNode(map);
  }, []);

  return (
    <Canvas
      camera={{ fov: 55, near: 0.1, far: 2000, position: [80, 60, 100] }}
      gl={{ antialias: true, powerPreference: "high-performance" }}
      style={{ position: "absolute", inset: 0 }}
      onCreated={({ gl }) => {
        gl.setClearColor(0x06060c);
        gl.setPixelRatio(Math.min(window.devicePixelRatio, 2));
      }}
    >
      <CameraController onDistanceChange={onDistanceChange} />

      <NodeRenderer
        ref={nodeMeshRef}
        nodeMap={nodeMap}
        layout={layout}
        lod={lod}
        filter={filter}
        onInstanceMapUpdate={handleInstanceMapUpdate}
      />

      <GlowRenderer nodeMap={nodeMap} layout={layout} lod={lod} filter={filter} />

      <EdgeRenderer
        nodeMap={nodeMap}
        edges={edges}
        layout={layout}
        lod={lod}
        filter={filter}
      />

      <LabelOverlay
        nodeMap={nodeMap}
        layout={layout}
        lod={lod}
        filter={filter}
        hoveredNodeId={hoveredNodeId}
      />

      <Picker
        meshRef={nodeMeshRef}
        instanceToNode={instanceToNode}
        nodeMap={nodeMap}
        onHover={onHover}
        onClick={onClick}
      />
    </Canvas>
  );
}
