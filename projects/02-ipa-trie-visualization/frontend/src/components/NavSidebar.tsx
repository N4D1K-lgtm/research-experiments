import { useViewStore, type StudyMode } from "../store/viewStore";
import { COLORS, FONTS } from "../styles/theme";

const NAV_ITEMS: { mode: StudyMode; label: string; icon: string; description: string }[] = [
  { mode: "explorer", label: "Explorer", icon: "◎", description: "3D trie visualization" },
  { mode: "analysis", label: "Analysis", icon: "◈", description: "Entropy & depth charts" },
  { mode: "phonology", label: "Phonology", icon: "◇", description: "IPA feature space" },
  { mode: "crossling", label: "Languages", icon: "◆", description: "Cross-linguistic comparison" },
];

export function NavSidebar() {
  const { mode, setMode } = useViewStore();

  return (
    <nav
      style={{
        position: "fixed",
        left: 0,
        top: 0,
        bottom: 0,
        width: 56,
        background: "rgba(6, 6, 12, 0.98)",
        borderRight: `1px solid ${COLORS.border}`,
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        paddingTop: 16,
        gap: 4,
        zIndex: 80,
        fontFamily: FONTS.sans,
      }}
    >
      {/* Logo / Title */}
      <div
        style={{
          width: 32,
          height: 32,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          marginBottom: 16,
          fontSize: 16,
          fontWeight: 600,
          color: COLORS.accent,
          fontFamily: FONTS.mono,
          background: `${COLORS.accent}10`,
          borderRadius: 8,
          border: `1px solid ${COLORS.accent}20`,
        }}
      >
        /
      </div>

      {NAV_ITEMS.map((item) => {
        const active = mode === item.mode;
        return (
          <button
            key={item.mode}
            onClick={() => setMode(item.mode)}
            title={`${item.label} — ${item.description}`}
            style={{
              width: 40,
              height: 40,
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              justifyContent: "center",
              gap: 2,
              background: active ? "rgba(255,255,255,0.08)" : "transparent",
              border: "none",
              borderRadius: 8,
              cursor: "pointer",
              color: active ? COLORS.textBright : COLORS.textDim,
              fontSize: 16,
              transition: "all 0.15s ease",
              position: "relative",
            }}
          >
            <span style={{ fontSize: 16, lineHeight: 1 }}>{item.icon}</span>
            <span
              style={{
                fontSize: 7,
                fontWeight: 500,
                textTransform: "uppercase",
                letterSpacing: 0.3,
                opacity: active ? 1 : 0.5,
              }}
            >
              {item.label}
            </span>
            {active && (
              <div
                style={{
                  position: "absolute",
                  left: -1,
                  top: 8,
                  bottom: 8,
                  width: 2,
                  background: COLORS.accent,
                  borderRadius: 1,
                }}
              />
            )}
          </button>
        );
      })}

      {/* Bottom spacer + stats link */}
      <div style={{ flex: 1 }} />
      <div
        style={{
          width: 32,
          height: 1,
          background: COLORS.border,
          marginBottom: 8,
        }}
      />
      <div
        style={{
          fontSize: 7,
          color: COLORS.textFaint,
          fontFamily: FONTS.mono,
          textAlign: "center",
          paddingBottom: 12,
          lineHeight: 1.4,
        }}
      >
        IPA
        <br />
        TRIE
      </div>
    </nav>
  );
}
