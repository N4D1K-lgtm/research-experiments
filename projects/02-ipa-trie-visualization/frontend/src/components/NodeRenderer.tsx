import { useRef, useMemo, useEffect, forwardRef, useImperativeHandle } from "react";
import * as THREE from "three";
import type { RenderNode, FilterState } from "../types/trie";
import type { RadialLayout } from "../utils/radialLayout";
import type { LODState } from "../utils/lod";
import { isNodeVisible, getNodeAlpha } from "../utils/lod";
import { blendNodeColor, hslToHex } from "../utils/colorBlender";
import { nodeEntropy, incomingProb } from "../utils/entropy";

const MIN_SCALE = 0.12;
const MAX_SCALE = 1.8;

interface Props {
  nodeMap: Map<number, RenderNode>;
  layout: RadialLayout | null;
  lod: LODState;
  filter: FilterState;
  onInstanceMapUpdate: (map: number[]) => void;
}

export const NodeRenderer = forwardRef<THREE.InstancedMesh, Props>(function NodeRenderer(
  { nodeMap, layout, lod, filter, onInstanceMapUpdate },
  ref,
) {
  const meshRef = useRef<THREE.InstancedMesh>(null);

  useImperativeHandle(ref, () => meshRef.current!, []);
  const maxCount = Math.max(nodeMap.size, 1000);

  const geo = useMemo(() => new THREE.IcosahedronGeometry(1, 2), []);
  const mat = useMemo(
    () => new THREE.MeshBasicMaterial({ toneMapped: false }),
    [],
  );

  useEffect(() => {
    if (!meshRef.current || !layout) return;

    const mesh = meshRef.current;
    const dummy = new THREE.Object3D();
    const color = new THREE.Color();
    const instanceToNode: number[] = [];
    let idx = 0;

    for (const [nodeId, layoutNode] of layout.nodes) {
      if (nodeId === 0) continue;
      const node = nodeMap.get(nodeId);
      if (!node) continue;
      if (node.depth > filter.maxDepth) continue;
      if (filter.terminalsOnly && !node.isTerminal) continue;
      if (filter.positionFilter.size > 0 && !filter.positionFilter.has(node.phonologicalPosition))
        continue;
      if (node.totalCount < filter.minFrequency || node.totalCount === 0) continue;
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

      if (node.isTerminal) {
        s = Math.min(MAX_SCALE, s * 1.4);
      } else if (entropy === 0 && !node.isTerminal) {
        s *= 0.5;
      }

      dummy.scale.setScalar(s);
      dummy.updateMatrix();
      mesh.setMatrixAt(idx, dummy.matrix);

      // Color
      const hsl = blendNodeColor(node);
      let l = hsl.l;
      let sat = hsl.s;

      if (entropy > 2.5) {
        l = Math.min(0.78, l + 0.08);
        sat = Math.max(0.25, sat - 0.1);
      } else if (entropy < 1.0 && node.childCount > 0) {
        sat = Math.min(1.0, sat + 0.15);
      }

      if (inProb > 0.15) {
        l = Math.min(0.82, l + inProb * 0.2);
      }

      const isMotifHighlighted =
        filter.highlightMotifs.size > 0 &&
        node.motifs?.some((m) => filter.highlightMotifs.has(m));

      if (node.isTerminal) {
        l = Math.min(0.88, l + 0.12);
        sat = Math.min(1.0, sat + 0.08);
      } else {
        l = Math.max(0.12, l * 0.55);
        sat = Math.max(0.08, sat * 0.55);
      }

      if (isMotifHighlighted) {
        l = Math.min(0.9, l + 0.2);
        sat = Math.min(1.0, sat + 0.2);
      } else if (filter.highlightMotifs.size > 0) {
        l *= 0.3;
        sat *= 0.3;
      }

      l *= alpha;
      color.setHex(hslToHex(hsl.h, sat, l));
      mesh.setColorAt(idx, color);

      instanceToNode.push(node.id);
      idx++;
    }

    mesh.count = idx;
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;

    onInstanceMapUpdate(instanceToNode);
  }, [nodeMap, layout, lod, filter, onInstanceMapUpdate]);

  return (
    <instancedMesh
      ref={meshRef}
      args={[geo, mat, maxCount]}
      frustumCulled={false}
    />
  );
});
