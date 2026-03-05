/** Shared design tokens matching the essay's visual language */

export const COLORS = {
  bg: "#0a0a12",
  bgPanel: "rgba(6, 6, 12, 0.96)",
  bgCard: "rgba(255,255,255,0.03)",
  bgCardHover: "rgba(255,255,255,0.06)",
  accent: "#4ECDC4",
  red: "#FF6B6B",
  yellow: "#FFE66D",
  purple: "#C49BFF",
  blue: "#45B7D1",
  green: "#96CEB4",
  onset: "#39a5c9",
  nucleus: "#c9a539",
  coda: "#9539c9",
  mixed: "#39c980",
  text: "#d0d0d8",
  textBright: "#e8e8f0",
  textDim: "#6a6a7a",
  textFaint: "#3a3a4a",
  border: "rgba(255,255,255,0.06)",
  borderLight: "rgba(255,255,255,0.08)",
  borderAccent: "rgba(78,205,196,0.2)",
} as const;

export const LANG_COLORS: Record<string, string> = {
  en_US: "#4ECDC4",
  fr_FR: "#FF6B6B",
  es_ES: "#FFE66D",
  de: "#C49BFF",
  nl: "#45B7D1",
  cmn: "#FF9F43",
  jpn: "#EE5A24",
  ara: "#A3CB38",
  fin: "#D980FA",
  tur: "#FDA7DF",
  hin: "#7EFFF5",
  swa: "#C4E538",
};

export const LANG_LABELS: Record<string, string> = {
  en_US: "English",
  fr_FR: "French",
  es_ES: "Spanish",
  de: "German",
  nl: "Dutch",
  cmn: "Mandarin",
  jpn: "Japanese",
  ara: "Arabic",
  fin: "Finnish",
  tur: "Turkish",
  hin: "Hindi",
  swa: "Swahili",
};

export function langColor(code: string): string {
  return LANG_COLORS[code] ?? "#888";
}

export function langLabel(code: string): string {
  return LANG_LABELS[code] ?? code;
}

export function hexAlpha(hex: string, alpha: number): string {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r},${g},${b},${Math.max(0, Math.min(1, alpha))})`;
}

export const FONTS = {
  sans: "'Inter', system-ui, sans-serif",
  mono: "'IBM Plex Mono', 'JetBrains Mono', 'Fira Code', monospace",
} as const;
