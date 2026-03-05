import { COLORS, FONTS } from "../../styles/theme";

/** Full-width chapter section wrapper */
export function Chapter({
  id,
  children,
  dark,
}: {
  id: string;
  children: React.ReactNode;
  dark?: boolean;
}) {
  return (
    <section
      id={id}
      style={{
        minHeight: "100vh",
        padding: "80px 0",
        background: dark ? "rgba(0,0,0,0.3)" : "transparent",
        position: "relative",
      }}
    >
      <div style={{ maxWidth: 760, margin: "0 auto", padding: "0 40px" }}>
        {children}
      </div>
    </section>
  );
}

/** Chapter number + title */
export function ChapterTitle({
  number,
  title,
  subtitle,
}: {
  number: number;
  title: string;
  subtitle?: string;
}) {
  return (
    <div style={{ marginBottom: 40 }}>
      <div
        style={{
          fontSize: 11,
          fontFamily: FONTS.mono,
          color: COLORS.accent,
          textTransform: "uppercase",
          letterSpacing: 2,
          marginBottom: 8,
        }}
      >
        Chapter {number}
      </div>
      <h2
        style={{
          fontSize: 32,
          fontWeight: 600,
          fontFamily: FONTS.mono,
          color: COLORS.textBright,
          lineHeight: 1.2,
          margin: 0,
        }}
      >
        {title}
      </h2>
      {subtitle && (
        <p
          style={{
            fontSize: 15,
            color: COLORS.textDim,
            lineHeight: 1.7,
            marginTop: 12,
            maxWidth: 560,
          }}
        >
          {subtitle}
        </p>
      )}
    </div>
  );
}

/** Prose paragraph */
export function Prose({ children }: { children: React.ReactNode }) {
  return (
    <p
      style={{
        fontSize: 15,
        lineHeight: 1.85,
        color: "#b0b0bc",
        marginBottom: 24,
        fontFamily: FONTS.sans,
      }}
    >
      {children}
    </p>
  );
}

/** Inline math / monospace term */
export function M({ children }: { children: React.ReactNode }) {
  return (
    <span
      style={{
        fontFamily: FONTS.mono,
        fontSize: "0.9em",
        color: COLORS.accent,
        background: `${COLORS.accent}0a`,
        padding: "1px 5px",
        borderRadius: 4,
        border: `1px solid ${COLORS.accent}15`,
      }}
    >
      {children}
    </span>
  );
}

/** Block math / formula display */
export function MathBlock({ children, label }: { children: React.ReactNode; label?: string }) {
  return (
    <div
      style={{
        background: "rgba(78, 205, 196, 0.04)",
        border: `1px solid ${COLORS.accent}18`,
        borderRadius: 12,
        padding: "20px 24px",
        margin: "24px 0",
        fontFamily: FONTS.mono,
        fontSize: 16,
        color: COLORS.textBright,
        textAlign: "center",
        lineHeight: 1.8,
        position: "relative",
      }}
    >
      {children}
      {label && (
        <div
          style={{
            position: "absolute",
            top: 8,
            right: 12,
            fontSize: 9,
            color: COLORS.textFaint,
            textTransform: "uppercase",
            letterSpacing: 1,
          }}
        >
          {label}
        </div>
      )}
    </div>
  );
}

/** Key insight callout box */
export function Insight({ children }: { children: React.ReactNode }) {
  return (
    <div
      style={{
        borderLeft: `3px solid ${COLORS.accent}`,
        padding: "14px 20px",
        margin: "24px 0",
        background: `${COLORS.accent}06`,
        borderRadius: "0 8px 8px 0",
        fontSize: 14,
        lineHeight: 1.7,
        color: COLORS.text,
        fontFamily: FONTS.sans,
      }}
    >
      {children}
    </div>
  );
}

/** Definition block */
export function Definition({ term, children }: { term: string; children: React.ReactNode }) {
  return (
    <div
      style={{
        background: "rgba(255,255,255,0.02)",
        border: `1px solid ${COLORS.border}`,
        borderRadius: 10,
        padding: "14px 18px",
        margin: "16px 0",
      }}
    >
      <div
        style={{
          fontSize: 11,
          fontFamily: FONTS.mono,
          fontWeight: 600,
          color: COLORS.accent,
          textTransform: "uppercase",
          letterSpacing: 0.8,
          marginBottom: 6,
        }}
      >
        {term}
      </div>
      <div style={{ fontSize: 13, lineHeight: 1.7, color: "#a0a0ac", fontFamily: FONTS.sans }}>
        {children}
      </div>
    </div>
  );
}

/** Interactive widget container */
export function Widget({
  children,
  label,
  instructions,
}: {
  children: React.ReactNode;
  label?: string;
  instructions?: string;
}) {
  return (
    <div
      style={{
        background: "rgba(255,255,255,0.02)",
        border: `1px solid ${COLORS.border}`,
        borderRadius: 14,
        padding: 24,
        margin: "28px 0",
        position: "relative",
      }}
    >
      {label && (
        <div
          style={{
            position: "absolute",
            top: -10,
            left: 16,
            padding: "2px 10px",
            background: COLORS.bg,
            border: `1px solid ${COLORS.border}`,
            borderRadius: 6,
            fontSize: 9,
            fontFamily: FONTS.mono,
            color: COLORS.textDim,
            textTransform: "uppercase",
            letterSpacing: 1,
          }}
        >
          {label}
        </div>
      )}
      {children}
      {instructions && (
        <div
          style={{
            marginTop: 12,
            fontSize: 10,
            color: COLORS.textFaint,
            fontFamily: FONTS.mono,
            fontStyle: "italic",
          }}
        >
          {instructions}
        </div>
      )}
    </div>
  );
}

/** Small colored tag */
export function Tag({ children, color }: { children: React.ReactNode; color: string }) {
  return (
    <span
      style={{
        display: "inline-block",
        padding: "2px 8px",
        background: `${color}15`,
        border: `1px solid ${color}25`,
        borderRadius: 5,
        fontSize: 11,
        fontFamily: FONTS.mono,
        fontWeight: 500,
        color,
      }}
    >
      {children}
    </span>
  );
}

/** Divider between sub-sections */
export function Divider() {
  return (
    <div
      style={{
        width: 40,
        height: 1,
        background: COLORS.border,
        margin: "40px 0",
      }}
    />
  );
}
