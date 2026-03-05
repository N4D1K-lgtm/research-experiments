import { useTrieDataStore } from "../../../store/trieDataStore";
import { COLORS, FONTS } from "../../../styles/theme";

export function ChapterHero() {
  const metadata = useTrieDataStore((s) => s.metadata);

  return (
    <section
      id="hero"
      style={{
        minHeight: "100vh",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        textAlign: "center",
        padding: "0 40px",
        position: "relative",
      }}
    >
      {/* Subtle radial glow */}
      <div
        style={{
          position: "absolute",
          top: "50%",
          left: "50%",
          transform: "translate(-50%, -50%)",
          width: 600,
          height: 600,
          background: `radial-gradient(circle, ${COLORS.accent}06 0%, transparent 70%)`,
          pointerEvents: "none",
        }}
      />

      <div
        style={{
          fontSize: 12,
          fontFamily: FONTS.mono,
          color: COLORS.accent,
          textTransform: "uppercase",
          letterSpacing: 4,
          marginBottom: 24,
          opacity: 0.6,
        }}
      >
        An Interactive Study
      </div>

      <h1
        style={{
          fontSize: 48,
          fontWeight: 600,
          fontFamily: FONTS.mono,
          color: COLORS.textBright,
          lineHeight: 1.15,
          margin: 0,
          maxWidth: 640,
        }}
      >
        The Shape of
        <br />
        <span style={{ color: COLORS.accent }}>Sound</span>
      </h1>

      <p
        style={{
          fontSize: 16,
          color: COLORS.textDim,
          lineHeight: 1.8,
          maxWidth: 520,
          marginTop: 24,
          fontFamily: FONTS.sans,
        }}
      >
        How {metadata?.totalWords?.toLocaleString() ?? "269,000+"} words from{" "}
        {metadata?.languages?.length ?? 12} languages compress into a single tree structure —
        and what that tree reveals about the hidden geometry of human speech.
      </p>

      {metadata && (
        <div
          style={{
            display: "flex",
            gap: 32,
            marginTop: 40,
            fontFamily: FONTS.mono,
          }}
        >
          <HeroStat value={metadata.nodeCount.toLocaleString()} label="nodes" />
          <HeroStat value={String(metadata.languages.length)} label="languages" />
          <HeroStat value={String(metadata.phonemeInventory.length)} label="phonemes" />
          <HeroStat value={String(metadata.maxDepth)} label="max depth" />
        </div>
      )}

      <div
        style={{
          marginTop: 60,
          fontSize: 11,
          color: COLORS.textFaint,
          fontFamily: FONTS.mono,
          animation: "pulse 2s ease-in-out infinite",
        }}
      >
        scroll to begin ↓
      </div>

      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 0.3; }
          50% { opacity: 0.7; }
        }
      `}</style>
    </section>
  );
}

function HeroStat({ value, label }: { value: string; label: string }) {
  return (
    <div>
      <div style={{ fontSize: 24, fontWeight: 600, color: COLORS.textBright }}>{value}</div>
      <div style={{ fontSize: 9, color: COLORS.textFaint, textTransform: "uppercase", letterSpacing: 1, marginTop: 2 }}>
        {label}
      </div>
    </div>
  );
}
