import { useRef, useCallback, useEffect } from "react";
import { useThree, useFrame } from "@react-three/fiber";
import * as THREE from "three";
import type { RenderNode } from "../types/trie";

interface Props {
  meshRef: React.RefObject<THREE.InstancedMesh | null>;
  instanceToNode: number[];
  nodeMap: Map<number, RenderNode>;
  onHover: (node: RenderNode | null, x: number, y: number) => void;
  onClick?: (node: RenderNode | null) => void;
}

export function Picker({ meshRef, instanceToNode, nodeMap, onHover, onClick }: Props) {
  const { camera, gl } = useThree();
  const raycaster = useRef(new THREE.Raycaster());
  const mouse = useRef(new THREE.Vector2());
  const lastInstanceId = useRef(-1);
  const mousePos = useRef({ x: 0, y: 0 });
  const lastHoveredNode = useRef<RenderNode | null>(null);

  // Track mouse position
  const handleMove = useCallback(
    (e: PointerEvent) => {
      const rect = gl.domElement.getBoundingClientRect();
      mouse.current.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      mouse.current.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
      mousePos.current = { x: e.clientX, y: e.clientY };
    },
    [gl],
  );

  // Click handler
  useEffect(() => {
    if (!onClick) return;
    const canvas = gl.domElement;
    const handleClick = () => {
      if (lastHoveredNode.current) {
        onClick(lastHoveredNode.current);
      }
    };
    canvas.addEventListener("click", handleClick);
    return () => canvas.removeEventListener("click", handleClick);
  }, [gl, onClick]);

  // Attach listener
  useFrame(() => {
    const canvas = gl.domElement;
    canvas.removeEventListener("pointermove", handleMove);
    canvas.addEventListener("pointermove", handleMove);
  });

  // Raycast each frame (throttled)
  const frameCount = useRef(0);
  useFrame(() => {
    frameCount.current++;
    if (frameCount.current % 3 !== 0) return;
    if (!meshRef.current) return;

    raycaster.current.setFromCamera(mouse.current, camera);
    const intersects = raycaster.current.intersectObject(meshRef.current);

    if (intersects.length > 0) {
      const instanceId = intersects[0].instanceId;
      if (instanceId != null && instanceId !== lastInstanceId.current) {
        lastInstanceId.current = instanceId;
        const nodeId = instanceToNode[instanceId];
        const node = nodeId != null ? nodeMap.get(nodeId) ?? null : null;
        lastHoveredNode.current = node;
        onHover(node, mousePos.current.x, mousePos.current.y);
      }
    } else if (lastInstanceId.current !== -1) {
      lastInstanceId.current = -1;
      lastHoveredNode.current = null;
      onHover(null, 0, 0);
    }
  });

  return null;
}
