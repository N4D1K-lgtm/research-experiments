import { useRef, useMemo, useEffect } from "react";
import * as THREE from "three";
import type { RenderNode, FilterState } from "../types/trie";
import type { RadialLayout } from "../utils/radialLayout";
import type { LODState } from "../utils/lod";
import { isNodeVisible, getNodeAlpha } from "../utils/lod";
import { blendNodeColor, hslToHex } from "../utils/colorBlender";
import { nodeEntropy, incomingProb } from "../utils/entropy";

const MIN_SCALE = 0.12;
const MAX_SCALE = 1.8;
const GLOW_SCALE = 2.5;

interface Props {
  nodeMap: Map<number, RenderNode>;
  layout: RadialLayout | null;
  lod: LODState;
  filter: FilterState;
}

export function GlowRenderer({ nodeMap, layout, lod, filter }: Props) {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const maxCount = Math.max(nodeMap.size, 500);

  const geo = useMemo(() => new THREE.IcosahedronGeometry(1, 1), []);
  const mat = useMemo(
    () =>
      new THREE.MeshBasicMaterial({
        transparent: true,
        opacity: 0.1,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
        toneMapped: false,
      }),
    [],
  );

  useEffect(() => {
    if (!meshRef.current || !layout) return;

    const mesh = meshRef.current;
    const dummy = new THREE.Object3D();
    const color = new THREE.Color();
    let idx = 0;

    for (const [nodeId, layoutNode] of layout.nodes) {
      if (nodeId === 0) continue;
      const node = nodeMap.get(nodeId);
      if (!node || !node.isTerminal) continue;
      if (node.depth > filter.maxDepth) continue;
      if (filter.terminalsOnly && !node.isTerminal) continue;
      if (filter.positionFilter.size > 0 && !filter.positionFilter.has(node.phonologicalPosition))
        continue;
      if (node.totalCount < filter.minFrequency) continue;
      if (!isNodeVisible(layoutNode, lod)) continue;
      const alpha = getNodeAlpha(layoutNode, lod);
      if (alpha <= 0) continue;

      dummy.position.set(layoutNode.x, layoutNode.y, layoutNode.z);

      const entropy = nodeEntropy(node);
      const inProb = incomingProb(node, nodeMap);
      const entropyNorm = Math.min(1, entropy / 4.3);
      const probNorm = Math.sqrt(Math.min(1, inProb * 3));

      let s =
        MIN_SCALE +
        (MAX_SCALE - MIN_SCALE) *
          (0.4 * entropyNorm +
            0.35 * probNorm +
            0.25 * (Math.log10(Math.max(node.totalCount, 1)) / 5));
      s = Math.min(MAX_SCALE, Math.max(MIN_SCALE, s));
      s = Math.min(MAX_SCALE, s * 1.4);

      const hsl = blendNodeColor(node);
      let l = Math.min(0.88, hsl.l + 0.12);
      let sat = Math.min(1.0, hsl.s + 0.08);
      l *= alpha;
      const glowL = Math.min(1.0, l * 1.5);

      dummy.scale.setScalar(s * GLOW_SCALE);
      dummy.updateMatrix();
      mesh.setMatrixAt(idx, dummy.matrix);

      color.setHex(hslToHex(hsl.h, sat, glowL));
      mesh.setColorAt(idx, color);
      idx++;
    }

    mesh.count = idx;
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  }, [nodeMap, layout, lod, filter]);

  return (
    <instancedMesh
      ref={meshRef}
      args={[geo, mat, maxCount]}
      frustumCulled={false}
    />
  );
}
