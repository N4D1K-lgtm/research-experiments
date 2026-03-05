import { useRef, useMemo, useEffect } from "react";
import * as THREE from "three";
import type { RenderNode, TrieEdge, FilterState } from "../types/trie";
import type { RadialLayout } from "../utils/radialLayout";
import type { LODState } from "../utils/lod";
import { isNodeVisible } from "../utils/lod";
import { blendNodeColor, hslToHex } from "../utils/colorBlender";

const HIGHWAY_THRESHOLD = 0.12;

interface Props {
  nodeMap: Map<number, RenderNode>;
  edges: TrieEdge[];
  layout: RadialLayout | null;
  lod: LODState;
  filter: FilterState;
}

export function EdgeRenderer({ nodeMap, edges, layout, lod }: Props) {
  const baseRef = useRef<THREE.LineSegments>(null);
  const hwRef = useRef<THREE.LineSegments>(null);

  const maxSegments = edges.length * 2 || 1000;
  const hwMaxSegments = Math.ceil(maxSegments * 0.6);

  const [baseGeo, baseMat] = useMemo(() => {
    const posArr = new Float32Array(maxSegments * 6);
    const colArr = new Float32Array(maxSegments * 6);
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(posArr, 3));
    geo.setAttribute("color", new THREE.BufferAttribute(colArr, 3));
    geo.setDrawRange(0, 0);
    const mat = new THREE.LineBasicMaterial({
      vertexColors: true,
      transparent: true,
      opacity: 0.12,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });
    return [geo, mat] as const;
  }, [maxSegments]);

  const [hwGeo, hwMat] = useMemo(() => {
    const posArr = new Float32Array(hwMaxSegments * 6);
    const colArr = new Float32Array(hwMaxSegments * 6);
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(posArr, 3));
    geo.setAttribute("color", new THREE.BufferAttribute(colArr, 3));
    geo.setDrawRange(0, 0);
    const mat = new THREE.LineBasicMaterial({
      vertexColors: true,
      transparent: true,
      opacity: 0.55,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });
    return [geo, mat] as const;
  }, [hwMaxSegments]);

  useEffect(() => {
    if (!layout) return;

    const posBuffer = baseGeo.attributes.position as THREE.BufferAttribute;
    const colorBuffer = baseGeo.attributes.color as THREE.BufferAttribute;
    const hwPosBuffer = hwGeo.attributes.position as THREE.BufferAttribute;
    const hwColorBuffer = hwGeo.attributes.color as THREE.BufferAttribute;

    const posArr = posBuffer.array as Float32Array;
    const colArr = colorBuffer.array as Float32Array;
    const hwPosArr = hwPosBuffer.array as Float32Array;
    const hwColArr = hwColorBuffer.array as Float32Array;

    let vertIdx = 0;
    let colorIdx = 0;
    let segmentCount = 0;
    let hwVertIdx = 0;
    let hwColorIdx = 0;
    let hwSegmentCount = 0;

    const tmpColor = new THREE.Color();

    for (const edge of edges) {
      if (segmentCount >= maxSegments) break;

      const srcLayout = layout.nodes.get(edge.source);
      const tgtLayout = layout.nodes.get(edge.target);
      if (!srcLayout || !tgtLayout) continue;
      if (!isNodeVisible(srcLayout, lod)) continue;
      if (!isNodeVisible(tgtLayout, lod)) continue;

      const cx = tgtLayout.x;
      const cy = tgtLayout.y;
      const cz = tgtLayout.z;
      const cLen = Math.sqrt(cx * cx + cy * cy + cz * cz);
      const invCLen = cLen > 0 ? 1 / cLen : 0;
      const dirX = cx * invCLen;
      const dirY = cy * invCLen;
      const dirZ = cz * invCLen;

      const parentR = srcLayout.sphereRadius;
      const edgeLen = tgtLayout.sphereRadius - parentR;
      const bendT = Math.min(0.4, Math.max(0, (edgeLen - 20) / 200));
      const bendR = parentR + edgeLen * bendT;
      const midX = dirX * bendR;
      const midY = dirY * bendR;
      const midZ = dirZ * bendR;

      const tgtNode = nodeMap.get(edge.target);
      const srcNode = nodeMap.get(edge.source);

      let transitionProb = 0.5;
      if (srcNode?.transitionProbs && tgtNode) {
        transitionProb = srcNode.transitionProbs[tgtNode.phoneme] ?? 0.05;
      }

      const probScale = Math.sqrt(Math.min(1, transitionProb * 4));

      if (tgtNode) {
        const hsl = blendNodeColor(tgtNode);
        tmpColor.setHex(hslToHex(hsl.h, hsl.s, hsl.l * 0.35 * probScale));
      } else {
        tmpColor.setRGB(0.05, 0.05, 0.05);
      }
      const cr = tmpColor.r;
      const cg = tmpColor.g;
      const cb = tmpColor.b;

      // Segment 1: parent → mid
      posArr[vertIdx++] = srcLayout.x;
      posArr[vertIdx++] = srcLayout.y;
      posArr[vertIdx++] = srcLayout.z;
      posArr[vertIdx++] = midX;
      posArr[vertIdx++] = midY;
      posArr[vertIdx++] = midZ;

      colArr[colorIdx++] = cr * 0.5;
      colArr[colorIdx++] = cg * 0.5;
      colArr[colorIdx++] = cb * 0.5;
      colArr[colorIdx++] = cr;
      colArr[colorIdx++] = cg;
      colArr[colorIdx++] = cb;

      // Segment 2: mid → child
      posArr[vertIdx++] = midX;
      posArr[vertIdx++] = midY;
      posArr[vertIdx++] = midZ;
      posArr[vertIdx++] = tgtLayout.x;
      posArr[vertIdx++] = tgtLayout.y;
      posArr[vertIdx++] = tgtLayout.z;

      colArr[colorIdx++] = cr;
      colArr[colorIdx++] = cg;
      colArr[colorIdx++] = cb;
      colArr[colorIdx++] = cr;
      colArr[colorIdx++] = cg;
      colArr[colorIdx++] = cb;

      segmentCount += 2;

      // Highway layer
      if (transitionProb >= HIGHWAY_THRESHOLD && hwSegmentCount + 2 <= hwMaxSegments) {
        const hwBright = Math.min(
          1,
          ((transitionProb - HIGHWAY_THRESHOLD) / 0.25) * 0.7 + 0.3,
        );

        if (tgtNode) {
          const hsl = blendNodeColor(tgtNode);
          tmpColor.setHex(
            hslToHex(
              hsl.h,
              Math.min(1, hsl.s + 0.15),
              Math.min(0.8, hsl.l * 0.7 * hwBright),
            ),
          );
        }
        const hr = tmpColor.r;
        const hg = tmpColor.g;
        const hb = tmpColor.b;

        hwPosArr[hwVertIdx++] = srcLayout.x;
        hwPosArr[hwVertIdx++] = srcLayout.y;
        hwPosArr[hwVertIdx++] = srcLayout.z;
        hwPosArr[hwVertIdx++] = midX;
        hwPosArr[hwVertIdx++] = midY;
        hwPosArr[hwVertIdx++] = midZ;

        hwColArr[hwColorIdx++] = hr * 0.6;
        hwColArr[hwColorIdx++] = hg * 0.6;
        hwColArr[hwColorIdx++] = hb * 0.6;
        hwColArr[hwColorIdx++] = hr;
        hwColArr[hwColorIdx++] = hg;
        hwColArr[hwColorIdx++] = hb;

        hwPosArr[hwVertIdx++] = midX;
        hwPosArr[hwVertIdx++] = midY;
        hwPosArr[hwVertIdx++] = midZ;
        hwPosArr[hwVertIdx++] = tgtLayout.x;
        hwPosArr[hwVertIdx++] = tgtLayout.y;
        hwPosArr[hwVertIdx++] = tgtLayout.z;

        hwColArr[hwColorIdx++] = hr;
        hwColArr[hwColorIdx++] = hg;
        hwColArr[hwColorIdx++] = hb;
        hwColArr[hwColorIdx++] = hr;
        hwColArr[hwColorIdx++] = hg;
        hwColArr[hwColorIdx++] = hb;

        hwSegmentCount += 2;
      }
    }

    posBuffer.needsUpdate = true;
    colorBuffer.needsUpdate = true;
    baseGeo.setDrawRange(0, segmentCount * 2);

    hwPosBuffer.needsUpdate = true;
    hwColorBuffer.needsUpdate = true;
    hwGeo.setDrawRange(0, hwSegmentCount * 2);
  }, [nodeMap, edges, layout, lod, baseGeo, hwGeo, maxSegments, hwMaxSegments]);

  return (
    <>
      <lineSegments ref={baseRef} geometry={baseGeo} material={baseMat} frustumCulled={false} />
      <lineSegments ref={hwRef} geometry={hwGeo} material={hwMat} frustumCulled={false} />
    </>
  );
}
