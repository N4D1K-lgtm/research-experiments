import type { TrieMetadata } from "../types/trie";

interface Props {
  metadata: TrieMetadata | null;
  visibleNodes: number;
}

export function StatsBar({ metadata, visibleNodes }: Props) {
  if (!metadata) return null;

  return (
    <div
      style={{
        position: "fixed",
        bottom: 20,
        left: 20,
        fontSize: 10,
        color: "#333",
        zIndex: 50,
        letterSpacing: 0.3,
        fontFamily: "'Inter', sans-serif",
      }}
    >
      {visibleNodes.toLocaleString()} nodes ·{" "}
      {metadata.totalWords.toLocaleString()} words ·{" "}
      {metadata.phonemeInventory.length} phonemes ·{" "}
      {metadata.languages.join(", ")}
    </div>
  );
}
