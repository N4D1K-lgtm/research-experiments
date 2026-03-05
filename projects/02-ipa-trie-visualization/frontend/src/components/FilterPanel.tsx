import type { TrieMetadata, PhonologicalPosition } from "../types/trie";
import { useFilterStore } from "../store/filterStore";
import { ROLES, getRoleColor } from "../utils/languagePalette";

interface Props {
  metadata: TrieMetadata;
}

export function FilterPanel({ metadata }: Props) {
  const {
    maxDepth,
    minFrequency,
    terminalsOnly,
    positionFilter,
    highlightMotifs,
    setMaxDepth,
    setMinFrequency,
    setTerminalsOnly,
    togglePosition,
    toggleMotif,
  } = useFilterStore();

  return (
    <div
      style={{
        position: "fixed",
        top: 20,
        left: 20,
        background: "rgba(8, 8, 16, 0.85)",
        border: "1px solid rgba(255, 255, 255, 0.06)",
        borderRadius: 14,
        padding: "18px 20px",
        zIndex: 50,
        minWidth: 210,
        backdropFilter: "blur(16px)",
        boxShadow: "0 4px 24px rgba(0,0,0,0.3)",
        fontFamily: "'Inter', sans-serif",
        color: "#d0d0d8",
        fontSize: 12,
      }}
    >
      {/* Phonological Role */}
      <Section title="Phonological Role">
        {ROLES.map((role) => (
          <CheckboxRow
            key={role.position}
            checked={positionFilter.has(role.position)}
            onChange={() => togglePosition(role.position)}
          >
            <span
              style={{
                display: "inline-block",
                width: 8,
                height: 8,
                borderRadius: "50%",
                background: getRoleColor(role.position),
                flexShrink: 0,
                boxShadow: `0 0 6px ${getRoleColor(role.position)}44`,
              }}
            />
            {role.label}
          </CheckboxRow>
        ))}
      </Section>

      {/* Display */}
      <Section title="Display">
        <CheckboxRow checked={terminalsOnly} onChange={() => setTerminalsOnly(!terminalsOnly)}>
          Word endpoints only
        </CheckboxRow>
      </Section>

      {/* Depth slider */}
      <Section title="Max Depth">
        <SliderRow
          min={1}
          max={metadata.maxDepth}
          value={maxDepth}
          onChange={setMaxDepth}
        />
      </Section>

      {/* Frequency slider */}
      <Section title="Min Frequency">
        <SliderRow min={1} max={5000} value={minFrequency} onChange={setMinFrequency} />
      </Section>

      {/* Motifs */}
      {metadata.motifs.length > 0 && (
        <Section title="Highlight Motif">
          {metadata.motifs.slice(0, 12).map((motif) => (
            <CheckboxRow
              key={motif.label}
              checked={highlightMotifs.has(motif.label)}
              onChange={() => toggleMotif(motif.label)}
            >
              <span style={{ fontSize: 11 }}>/{motif.label}/</span>
              <span
                style={{
                  fontSize: 9,
                  color: "#555",
                  marginLeft: "auto",
                  fontVariantNumeric: "tabular-nums",
                }}
              >
                {motif.count.toLocaleString()}
              </span>
            </CheckboxRow>
          ))}
        </Section>
      )}

      {/* Legend */}
      <div
        style={{
          marginTop: 14,
          paddingTop: 12,
          borderTop: "1px solid rgba(255,255,255,0.05)",
          fontSize: 9,
          color: "#444",
          lineHeight: 1.6,
        }}
      >
        <LegendRow color="#39a5c9">Onset (consonant)</LegendRow>
        <LegendRow color="#c9a539">Nucleus (vowel)</LegendRow>
        <LegendRow color="#9539c9">Coda (consonant)</LegendRow>
        <LegendRow color="#39c980">Mixed role</LegendRow>
        <LegendRow color="rgba(255,255,255,0.6)" style={{ marginTop: 4 }}>
          Glow = word endpoint
        </LegendRow>
        <LegendRow color="rgba(255,255,255,0.25)" height={6}>
          Large = high entropy
        </LegendRow>
      </div>
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div style={{ marginBottom: 16 }}>
      <h3
        style={{
          fontSize: 10,
          fontWeight: 600,
          marginBottom: 10,
          color: "#555",
          textTransform: "uppercase",
          letterSpacing: 1.2,
        }}
      >
        {title}
      </h3>
      {children}
    </div>
  );
}

function CheckboxRow({
  checked,
  onChange,
  children,
}: {
  checked: boolean;
  onChange: () => void;
  children: React.ReactNode;
}) {
  return (
    <label
      style={{
        display: "flex",
        alignItems: "center",
        gap: 8,
        cursor: "pointer",
        padding: "3px 0",
        fontSize: 12,
      }}
    >
      <input
        type="checkbox"
        checked={checked}
        onChange={onChange}
        style={{ accentColor: "#888" }}
      />
      {children}
    </label>
  );
}

function SliderRow({
  min,
  max,
  value,
  onChange,
}: {
  min: number;
  max: number;
  value: number;
  onChange: (v: number) => void;
}) {
  return (
    <div
      style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 11, color: "#555", marginTop: 4 }}
    >
      {min}
      <input
        type="range"
        min={min}
        max={max}
        value={value}
        onChange={(e) => onChange(parseInt(e.target.value))}
        style={{ flex: 1 }}
      />
      <span
        style={{ minWidth: 28, textAlign: "right", fontVariantNumeric: "tabular-nums", color: "#777" }}
      >
        {value}
      </span>
    </div>
  );
}

function LegendRow({
  color,
  height = 4,
  style,
  children,
}: {
  color: string;
  height?: number;
  style?: React.CSSProperties;
  children: React.ReactNode;
}) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 6, ...style }}>
      <div
        style={{
          width: 20,
          height,
          borderRadius: 2,
          background: color,
          flexShrink: 0,
        }}
      />
      <span>{children}</span>
    </div>
  );
}
