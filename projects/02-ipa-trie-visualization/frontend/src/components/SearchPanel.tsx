import { useState, useCallback } from "react";
import { useClient } from "urql";
import { SEARCH_QUERY } from "../graphql/queries";
import { toRenderNode, type RenderNode } from "../types/trie";
import { getRoleColor } from "../utils/languagePalette";

interface Props {
  onNodeSelect: (node: RenderNode) => void;
}

export function SearchPanel({ onNodeSelect }: Props) {
  const client = useClient();
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<RenderNode[]>([]);
  const [searching, setSearching] = useState(false);

  const handleSearch = useCallback(async () => {
    if (!query.trim()) return;
    setSearching(true);
    const result = await client
      .query(SEARCH_QUERY, { phoneme: query.trim(), limit: 20 })
      .toPromise();
    if (result.data?.search?.nodes) {
      setResults(result.data.search.nodes.map(toRenderNode));
    }
    setSearching(false);
  }, [client, query]);

  return (
    <div
      style={{
        position: "fixed",
        top: 20,
        right: 20,
        background: "rgba(8, 8, 16, 0.85)",
        border: "1px solid rgba(255, 255, 255, 0.06)",
        borderRadius: 14,
        padding: "12px 16px",
        zIndex: 50,
        minWidth: 200,
        maxWidth: 280,
        backdropFilter: "blur(16px)",
        boxShadow: "0 4px 24px rgba(0,0,0,0.3)",
        fontFamily: "'Inter', sans-serif",
        color: "#d0d0d8",
        fontSize: 12,
      }}
    >
      <div style={{ display: "flex", gap: 6 }}>
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleSearch()}
          placeholder="Search phoneme..."
          style={{
            flex: 1,
            background: "rgba(255,255,255,0.05)",
            border: "1px solid rgba(255,255,255,0.1)",
            borderRadius: 6,
            padding: "6px 10px",
            color: "#d0d0d8",
            fontSize: 12,
            outline: "none",
            fontFamily: "'Inter', sans-serif",
          }}
        />
        <button
          onClick={handleSearch}
          disabled={searching}
          style={{
            background: "rgba(255,255,255,0.08)",
            border: "1px solid rgba(255,255,255,0.1)",
            borderRadius: 6,
            padding: "6px 12px",
            color: "#888",
            cursor: "pointer",
            fontSize: 11,
          }}
        >
          {searching ? "..." : "Go"}
        </button>
      </div>

      {results.length > 0 && (
        <div style={{ marginTop: 8, maxHeight: 200, overflowY: "auto" }}>
          {results.map((node) => (
            <div
              key={node.id}
              onClick={() => onNodeSelect(node)}
              style={{
                padding: "4px 6px",
                cursor: "pointer",
                borderRadius: 4,
                display: "flex",
                gap: 8,
                alignItems: "center",
                fontSize: 11,
              }}
              onMouseEnter={(e) =>
                (e.currentTarget.style.background = "rgba(255,255,255,0.05)")
              }
              onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
            >
              <span
                style={{
                  color: getRoleColor(node.phonologicalPosition),
                  fontFamily: "monospace",
                  fontWeight: 600,
                }}
              >
                /{node.phoneme}/
              </span>
              <span style={{ color: "#555" }}>d{node.depth}</span>
              <span style={{ color: "#555", marginLeft: "auto" }}>
                {node.totalCount.toLocaleString()}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
