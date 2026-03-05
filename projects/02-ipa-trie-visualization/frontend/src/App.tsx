import { useTrieDataStore } from "./store/trieDataStore";
import { useTrieData } from "./hooks/useTrieData";
import { useLOD } from "./hooks/useLOD";
import { COLORS, FONTS } from "./styles/theme";
import { TutorialShell } from "./components/tutorial/TutorialShell";

function App() {
  const { metadata, error } = useTrieDataStore();
  const maxDepth = metadata?.maxDepth ?? 12;
  const { lod } = useLOD(maxDepth);

  // Progressive loading — start fetching trie data immediately
  useTrieData(lod.visibleMaxDepth);

  // Loading state
  if (!metadata && !error) {
    return (
      <div
        style={{
          position: "fixed",
          inset: 0,
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          background: COLORS.bg,
          gap: 16,
          fontFamily: FONTS.sans,
        }}
      >
        <div
          style={{
            width: 28,
            height: 28,
            border: "2px solid rgba(255,255,255,0.08)",
            borderTopColor: "rgba(255,255,255,0.4)",
            borderRadius: "50%",
            animation: "spin 0.8s linear infinite",
          }}
        />
        <span style={{ fontSize: 13, color: COLORS.textFaint, letterSpacing: 0.3 }}>
          Loading phonological trie...
        </span>
        <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
      </div>
    );
  }

  if (error) {
    return (
      <div
        style={{
          position: "fixed",
          inset: 0,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          background: COLORS.bg,
          color: "#c55",
          fontFamily: FONTS.sans,
        }}
      >
        Error: {error}
      </div>
    );
  }

  return (
    <>
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
      <TutorialShell />
    </>
  );
}

export default App;
