import type { TrieNodeData } from "../types";
import { getRoleHue } from "./languagePalette";

/**
 * Compute HSL color based on phonological role.
 *
 * Hue: determined by phonological position (onset=cool, nucleus=warm, coda=purple)
 * Saturation: higher for clear roles, lower for mixed
 * Lightness: log-scaled by frequency, boosted for terminals
 */
export function blendNodeColor(
  node: TrieNodeData,
): { h: number; s: number; l: number } {
  const total = node.totalCount;
  if (total === 0) {
    return { h: 0, s: 0, l: 0.1 };
  }

  const hue = getRoleHue(node.phonologicalPosition);
  const saturation = node.phonologicalPosition === "mixed" ? 0.35 : 0.65;
  const lightness = Math.min(0.72, 0.32 + 0.18 * Math.log10(Math.max(total, 1)));

  return { h: hue, s: saturation, l: lightness };
}

export function hslToHex(h: number, s: number, l: number): number {
  h /= 360;
  let r: number, g: number, b: number;
  if (s === 0) {
    r = g = b = l;
  } else {
    const hue2rgb = (p: number, q: number, t: number) => {
      if (t < 0) t += 1;
      if (t > 1) t -= 1;
      if (t < 1 / 6) return p + (q - p) * 6 * t;
      if (t < 1 / 2) return q;
      if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
      return p;
    };
    const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
    const p = 2 * l - q;
    r = hue2rgb(p, q, h + 1 / 3);
    g = hue2rgb(p, q, h);
    b = hue2rgb(p, q, h - 1 / 3);
  }
  return (
    ((Math.round(r * 255) & 0xff) << 16) |
    ((Math.round(g * 255) & 0xff) << 8) |
    (Math.round(b * 255) & 0xff)
  );
}
