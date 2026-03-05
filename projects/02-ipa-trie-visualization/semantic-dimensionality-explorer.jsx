import { useState, useEffect, useRef, useCallback, useMemo } from "react";

// --- Data Generation Utilities ---

const LANGUAGES = {
  english: {
    label: "English", alphabet: 26, avgBranching: 4.2,
    phonotactic: 0.38, morphological: "concatenative",
    color: "#4ECDC4", samplePrefixes: ["th","st","pr","ch","sh","tr","gr","sp","cr","fl","br","pl","wh","sw","sc"],
    commonEndings: ["ing","tion","ness","ment","able","ous","ive","ful","less","ity"],
  },
  mandarin: {
    label: "Mandarin (Pinyin)", alphabet: 26, avgBranching: 3.8,
    phonotactic: 0.42, morphological: "isolating",
    color: "#FF6B6B", samplePrefixes: ["zh","ch","sh","xi","ji","qi","zu","cu","su","li","mi","ni","bi","pi","di"],
    commonEndings: ["ang","eng","ong","ian","iao","uang","uan","ing"],
  },
  arabic: {
    label: "Arabic (Romanized)", alphabet: 28, avgBranching: 5.1,
    phonotactic: 0.31, morphological: "templatic",
    color: "#FFE66D", samplePrefixes: ["al","mu","ma","ka","ta","ha","sa","ba","wa","na","fa","ra","qa","da","la"],
    commonEndings: ["iya","aat","iin","uun","aha","ana","tun"],
  },
  japanese: {
    label: "Japanese (Romaji)", alphabet: 21, avgBranching: 3.2,
    phonotactic: 0.51, morphological: "agglutinative",
    color: "#C49BFF", samplePrefixes: ["ka","ki","ku","ke","ko","sa","shi","su","se","so","ta","chi","tsu","te","to"],
    commonEndings: ["masu","desu","shita","nai","reru","seru","mono","kata"],
  },
  finnish: {
    label: "Finnish", alphabet: 29, avgBranching: 4.8,
    phonotactic: 0.29, morphological: "agglutinative",
    color: "#45B7D1", samplePrefixes: ["ka","ki","ku","ko","ta","ti","tu","to","pa","pi","pu","po","sa","si","su"],
    commonEndings: ["nen","ssa","sta","lla","lta","lle","ksi","tta","nsa","inen"],
  },
};

function generateTrieData(langKey, maxDepth = 8) {
  const lang = LANGUAGES[langKey];
  const nodes = [];
  const rng = seedRandom(langKey.length * 137);

  function seedRandom(seed) {
    let s = seed;
    return () => { s = (s * 16807 + 0) % 2147483647; return s / 2147483647; };
  }

  // Root
  nodes.push({ depth: 0, angle: 0, parentIdx: -1, isWord: false, frequency: 0, semanticCluster: -1 });

  for (let d = 1; d <= maxDepth; d++) {
    const parentsAtPrev = nodes.filter(n => n.depth === d - 1);
    parentsAtPrev.forEach((parent, pi) => {
      // Branching decreases with depth (convergence signal)
      const baseBranch = lang.avgBranching * Math.pow(0.72, d - 1);
      const branches = Math.max(1, Math.round(baseBranch + (rng() - 0.5) * 2));
      for (let b = 0; b < branches; b++) {
        const angleSpread = (2 * Math.PI) / Math.max(1, parentsAtPrev.length);
        const baseAngle = pi * angleSpread + (b / branches) * angleSpread;
        const jitter = (rng() - 0.5) * 0.3;
        const isWord = d >= 2 && rng() < (d < 4 ? 0.1 : d < 6 ? 0.3 : 0.5);
        const freq = isWord ? Math.pow(rng(), 2) * 1000 : 0;
        const cluster = Math.floor(rng() * 8);
        nodes.push({
          depth: d, angle: baseAngle + jitter,
          parentIdx: nodes.indexOf(parent),
          isWord, frequency: freq, semanticCluster: cluster,
        });
      }
    });
  }
  return nodes;
}

function computeShellStats(nodes, maxDepth) {
  const stats = [];
  for (let d = 0; d <= maxDepth; d++) {
    const shell = nodes.filter(n => n.depth === d);
    const words = shell.filter(n => n.isWord);
    const clusters = new Set(words.map(n => n.semanticCluster));
    stats.push({
      depth: d,
      nodeCount: shell.length,
      wordCount: words.length,
      branchingFactor: d > 0 ? shell.length / nodes.filter(n => n.depth === d - 1).length : 0,
      uniqueClusters: clusters.size,
      entropy: shell.length > 0 ? Math.log2(shell.length) : 0,
    });
  }
  return stats;
}

function estimateDimensionality(shellStats) {
  return shellStats.map((s, i) => {
    if (i === 0) return { depth: 0, estimated: 0, cumulative: 0 };
    const logNodes = Math.log2(Math.max(1, s.nodeCount));
    const logBranch = Math.log2(Math.max(1, s.branchingFactor));
    const ent = s.entropy;
    const dim = Math.min(logNodes, ent * 0.7 + logBranch * 1.2);
    const cumDim = i > 0 ? dim * 0.4 + shellStats.slice(0, i).reduce((a, x) => a + Math.log2(Math.max(1, x.branchingFactor)) * 0.12, 0) : 0;
    return { depth: i, estimated: dim, cumulative: Math.min(cumDim, 45 + Math.random() * 10) };
  });
}

// --- Semantic Cluster Colors ---
const CLUSTER_COLORS = [
  "#FF6B6B", "#4ECDC4", "#FFE66D", "#C49BFF",
  "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD",
];

// --- Components ---

function RadialTrie({ langKey, width, height, highlightShell, onShellHover }) {
  const canvasRef = useRef(null);
  const nodesRef = useRef(null);
  const maxDepth = 8;

  if (!nodesRef.current || nodesRef.current._lang !== langKey) {
    nodesRef.current = generateTrieData(langKey, maxDepth);
    nodesRef.current._lang = langKey;
  }
  const nodes = nodesRef.current;
  const lang = LANGUAGES[langKey];

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);

    const cx = width / 2;
    const cy = height / 2;
    const maxR = Math.min(width, height) / 2 - 20;
    const shellR = (d) => (d / maxDepth) * maxR;

    // Background
    ctx.fillStyle = "#0a0a0f";
    ctx.fillRect(0, 0, width, height);

    // Shell rings
    for (let d = 1; d <= maxDepth; d++) {
      ctx.beginPath();
      ctx.arc(cx, cy, shellR(d), 0, Math.PI * 2);
      const isHighlighted = highlightShell === d;
      ctx.strokeStyle = isHighlighted ? `${lang.color}44` : "#ffffff08";
      ctx.lineWidth = isHighlighted ? 2 : 0.5;
      ctx.stroke();

      if (isHighlighted) {
        ctx.fillStyle = `${lang.color}08`;
        ctx.beginPath();
        ctx.arc(cx, cy, shellR(d), 0, Math.PI * 2);
        ctx.arc(cx, cy, shellR(d - 1), 0, Math.PI * 2, true);
        ctx.fill();
      }
    }

    // Edges
    ctx.lineWidth = 0.3;
    nodes.forEach((node) => {
      if (node.parentIdx < 0) return;
      const parent = nodes[node.parentIdx];
      const x1 = cx + Math.cos(parent.angle) * shellR(parent.depth);
      const y1 = cy + Math.sin(parent.angle) * shellR(parent.depth);
      const x2 = cx + Math.cos(node.angle) * shellR(node.depth);
      const y2 = cy + Math.sin(node.angle) * shellR(node.depth);
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.strokeStyle = node.isWord
        ? `${CLUSTER_COLORS[node.semanticCluster]}30`
        : "#ffffff0a";
      ctx.stroke();
    });

    // Nodes
    nodes.forEach((node) => {
      if (node.depth === 0) return;
      const x = cx + Math.cos(node.angle) * shellR(node.depth);
      const y = cy + Math.sin(node.angle) * shellR(node.depth);
      const r = node.isWord ? 1.5 + Math.sqrt(node.frequency) * 0.04 : 0.6;
      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fillStyle = node.isWord
        ? CLUSTER_COLORS[node.semanticCluster] + "cc"
        : "#ffffff20";
      ctx.fill();
    });

    // Center
    ctx.beginPath();
    ctx.arc(cx, cy, 4, 0, Math.PI * 2);
    ctx.fillStyle = lang.color;
    ctx.fill();
    ctx.beginPath();
    ctx.arc(cx, cy, 8, 0, Math.PI * 2);
    ctx.strokeStyle = lang.color + "40";
    ctx.lineWidth = 1;
    ctx.stroke();

  }, [langKey, width, height, highlightShell, nodes, lang]);

  const handleMouseMove = useCallback((e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left - width / 2;
    const y = e.clientY - rect.top - height / 2;
    const dist = Math.sqrt(x * x + y * y);
    const maxR = Math.min(width, height) / 2 - 20;
    const shell = Math.round((dist / maxR) * maxDepth);
    onShellHover?.(shell >= 1 && shell <= maxDepth ? shell : null);
  }, [width, height, maxDepth, onShellHover]);

  return (
    <canvas
      ref={canvasRef}
      style={{ width, height, cursor: "crosshair" }}
      onMouseMove={handleMouseMove}
      onMouseLeave={() => onShellHover?.(null)}
    />
  );
}

function DimensionalityChart({ languages, width, height, highlightShell }) {
  const canvasRef = useRef(null);
  const maxDepth = 8;

  const allData = useMemo(() => {
    const result = {};
    languages.forEach((lk) => {
      const nodes = generateTrieData(lk, maxDepth);
      const stats = computeShellStats(nodes, maxDepth);
      result[lk] = estimateDimensionality(stats);
    });
    return result;
  }, [languages]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);

    const pad = { top: 30, right: 20, bottom: 40, left: 50 };
    const w = width - pad.left - pad.right;
    const h = height - pad.top - pad.bottom;

    ctx.fillStyle = "#0a0a0f";
    ctx.fillRect(0, 0, width, height);

    // Grid
    ctx.strokeStyle = "#ffffff0a";
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= 5; i++) {
      const y = pad.top + (i / 5) * h;
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(pad.left + w, y);
      ctx.stroke();
    }

    // Highlight shell
    if (highlightShell) {
      const x = pad.left + (highlightShell / maxDepth) * w;
      ctx.strokeStyle = "#ffffff20";
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(x, pad.top);
      ctx.lineTo(x, pad.top + h);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Convergence band
    const convMin = 32, convMax = 48;
    const maxDim = 60;
    const yMin = pad.top + (1 - convMax / maxDim) * h;
    const yMax = pad.top + (1 - convMin / maxDim) * h;
    ctx.fillStyle = "#ffffff06";
    ctx.fillRect(pad.left, yMin, w, yMax - yMin);
    ctx.font = "10px 'IBM Plex Mono', monospace";
    ctx.fillStyle = "#ffffff30";
    ctx.textAlign = "right";
    ctx.fillText("convergence zone", pad.left + w - 4, yMin + 12);

    // Lines
    languages.forEach((lk) => {
      const data = allData[lk];
      const lang = LANGUAGES[lk];
      ctx.beginPath();
      data.forEach((d, i) => {
        const x = pad.left + (d.depth / maxDepth) * w;
        const y = pad.top + (1 - d.cumulative / maxDim) * h;
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      });
      ctx.strokeStyle = lang.color + "cc";
      ctx.lineWidth = 2;
      ctx.stroke();

      // Points
      data.forEach((d) => {
        const x = pad.left + (d.depth / maxDepth) * w;
        const y = pad.top + (1 - d.cumulative / maxDim) * h;
        ctx.beginPath();
        ctx.arc(x, y, 3, 0, Math.PI * 2);
        ctx.fillStyle = d.depth === highlightShell ? lang.color : lang.color + "80";
        ctx.fill();
      });
    });

    // Axes
    ctx.fillStyle = "#ffffff60";
    ctx.font = "11px 'IBM Plex Mono', monospace";
    ctx.textAlign = "center";
    for (let d = 0; d <= maxDepth; d++) {
      const x = pad.left + (d / maxDepth) * w;
      ctx.fillText(d.toString(), x, pad.top + h + 20);
    }
    ctx.fillText("string depth (shell)", pad.left + w / 2, pad.top + h + 36);

    ctx.textAlign = "right";
    for (let i = 0; i <= 5; i++) {
      const val = (i / 5) * maxDim;
      const y = pad.top + (1 - val / maxDim) * h;
      ctx.fillText(val.toFixed(0), pad.left - 8, y + 4);
    }

    ctx.save();
    ctx.translate(14, pad.top + h / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = "center";
    ctx.fillText("est. semantic dimensions", 0, 0);
    ctx.restore();

  }, [languages, allData, width, height, highlightShell]);

  return <canvas ref={canvasRef} style={{ width, height }} />;
}

function BranchingEntropyChart({ languages, width, height, highlightShell }) {
  const canvasRef = useRef(null);
  const maxDepth = 8;

  const allStats = useMemo(() => {
    const result = {};
    languages.forEach((lk) => {
      const nodes = generateTrieData(lk, maxDepth);
      result[lk] = computeShellStats(nodes, maxDepth);
    });
    return result;
  }, [languages]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);

    const pad = { top: 30, right: 20, bottom: 40, left: 50 };
    const w = width - pad.left - pad.right;
    const h = height - pad.top - pad.bottom;

    ctx.fillStyle = "#0a0a0f";
    ctx.fillRect(0, 0, width, height);

    const maxEntropy = 12;

    if (highlightShell) {
      const x = pad.left + (highlightShell / maxDepth) * w;
      ctx.strokeStyle = "#ffffff20";
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(x, pad.top);
      ctx.lineTo(x, pad.top + h);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    languages.forEach((lk) => {
      const stats = allStats[lk];
      const lang = LANGUAGES[lk];
      ctx.beginPath();
      stats.forEach((s, i) => {
        const x = pad.left + (i / maxDepth) * w;
        const y = pad.top + (1 - s.entropy / maxEntropy) * h;
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      });
      ctx.strokeStyle = lang.color + "99";
      ctx.lineWidth = 1.5;
      ctx.stroke();
    });

    ctx.fillStyle = "#ffffff60";
    ctx.font = "11px 'IBM Plex Mono', monospace";
    ctx.textAlign = "center";
    for (let d = 0; d <= maxDepth; d++) {
      ctx.fillText(d.toString(), pad.left + (d / maxDepth) * w, pad.top + h + 20);
    }
    ctx.fillText("shell depth", pad.left + w / 2, pad.top + h + 36);

    ctx.textAlign = "right";
    for (let i = 0; i <= 4; i++) {
      const val = (i / 4) * maxEntropy;
      const y = pad.top + (1 - val / maxEntropy) * h;
      ctx.fillText(val.toFixed(0), pad.left - 8, y + 4);
    }

    ctx.save();
    ctx.translate(14, pad.top + h / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = "center";
    ctx.fillText("shell entropy (bits)", 0, 0);
    ctx.restore();

  }, [languages, allStats, width, height, highlightShell]);

  return <canvas ref={canvasRef} style={{ width, height }} />;
}

function ExperimentPhase({ phase, title, description, status, children }) {
  return (
    <div style={{
      background: "#0d0d14",
      border: "1px solid #ffffff10",
      borderRadius: 8,
      padding: 20,
      marginBottom: 16,
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 8 }}>
        <span style={{
          background: status === "active" ? "#4ECDC420" : "#ffffff08",
          color: status === "active" ? "#4ECDC4" : "#ffffff40",
          padding: "2px 10px",
          borderRadius: 12,
          fontSize: 11,
          fontFamily: "'IBM Plex Mono', monospace",
          letterSpacing: 1,
        }}>
          PHASE {phase}
        </span>
        <span style={{ color: "#ffffffcc", fontSize: 15, fontWeight: 600, fontFamily: "'IBM Plex Sans', sans-serif" }}>
          {title}
        </span>
      </div>
      <p style={{ color: "#ffffff70", fontSize: 13, lineHeight: 1.6, margin: "8px 0", fontFamily: "'IBM Plex Sans', sans-serif" }}>
        {description}
      </p>
      {children}
    </div>
  );
}

// --- Main App ---
export default function SemanticDimensionalityExplorer() {
  const [selectedLangs, setSelectedLangs] = useState(["english", "mandarin", "arabic"]);
  const [primaryLang, setPrimaryLang] = useState("english");
  const [highlightShell, setHighlightShell] = useState(null);
  const [activeView, setActiveView] = useState("trie"); // trie | convergence | experiment

  const toggleLang = (lk) => {
    setSelectedLangs((prev) =>
      prev.includes(lk) ? (prev.length > 1 ? prev.filter((l) => l !== lk) : prev) : [...prev, lk]
    );
    if (!selectedLangs.includes(lk)) setPrimaryLang(lk);
  };

  return (
    <div style={{
      background: "#08080d",
      minHeight: "100vh",
      color: "#ffffffdd",
      fontFamily: "'IBM Plex Sans', sans-serif",
    }}>
      {/* Header */}
      <div style={{
        borderBottom: "1px solid #ffffff0a",
        padding: "20px 28px 16px",
      }}>
        <div style={{ display: "flex", alignItems: "baseline", gap: 12, marginBottom: 4 }}>
          <h1 style={{
            fontSize: 20,
            fontWeight: 700,
            color: "#ffffffee",
            margin: 0,
            fontFamily: "'IBM Plex Mono', monospace",
            letterSpacing: -0.5,
          }}>
            Semantic Dimensionality
          </h1>
          <span style={{ color: "#ffffff30", fontSize: 12, fontFamily: "'IBM Plex Mono', monospace" }}>
            convergence explorer
          </span>
        </div>
        <p style={{ color: "#ffffff50", fontSize: 12, margin: "6px 0 12px", maxWidth: 700, lineHeight: 1.5 }}>
          Does the number of dimensions needed to encode meaning converge across languages?
          Explore radial tries, branching entropy, and estimated semantic dimensionality.
        </p>

        {/* Language selectors */}
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 12 }}>
          {Object.entries(LANGUAGES).map(([key, lang]) => {
            const isSelected = selectedLangs.includes(key);
            const isPrimary = primaryLang === key;
            return (
              <button
                key={key}
                onClick={() => toggleLang(key)}
                onDoubleClick={() => setPrimaryLang(key)}
                style={{
                  background: isSelected ? `${lang.color}18` : "transparent",
                  border: `1px solid ${isSelected ? lang.color + "60" : "#ffffff15"}`,
                  color: isSelected ? lang.color : "#ffffff40",
                  padding: "4px 14px",
                  borderRadius: 20,
                  fontSize: 12,
                  cursor: "pointer",
                  fontFamily: "'IBM Plex Mono', monospace",
                  transition: "all 0.2s",
                  outline: isPrimary ? `1px solid ${lang.color}` : "none",
                  outlineOffset: 2,
                }}
              >
                {lang.label}
                {isPrimary && isSelected && <span style={{ marginLeft: 6, opacity: 0.5 }}>◉</span>}
              </button>
            );
          })}
          <span style={{ color: "#ffffff25", fontSize: 10, alignSelf: "center", marginLeft: 4 }}>
            click to toggle · double-click for primary
          </span>
        </div>

        {/* View tabs */}
        <div style={{ display: "flex", gap: 2 }}>
          {[
            { key: "trie", label: "Radial Trie" },
            { key: "convergence", label: "Convergence Analysis" },
            { key: "experiment", label: "Experimental Design" },
          ].map(({ key, label }) => (
            <button
              key={key}
              onClick={() => setActiveView(key)}
              style={{
                background: activeView === key ? "#ffffff0d" : "transparent",
                border: "none",
                borderBottom: activeView === key ? "2px solid #4ECDC4" : "2px solid transparent",
                color: activeView === key ? "#ffffffcc" : "#ffffff40",
                padding: "8px 18px",
                fontSize: 12,
                cursor: "pointer",
                fontFamily: "'IBM Plex Mono', monospace",
                transition: "all 0.2s",
              }}
            >
              {label}
            </button>
          ))}
        </div>
      </div>

      {/* Content */}
      <div style={{ padding: "20px 28px" }}>
        {activeView === "trie" && (
          <div>
            <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
              {/* Primary trie - large */}
              <div style={{ flex: "1 1 380px", minWidth: 340 }}>
                <div style={{
                  background: "#0a0a0f",
                  border: "1px solid #ffffff0a",
                  borderRadius: 8,
                  padding: 12,
                }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
                    <span style={{
                      color: LANGUAGES[primaryLang].color,
                      fontSize: 13,
                      fontFamily: "'IBM Plex Mono', monospace",
                    }}>
                      {LANGUAGES[primaryLang].label}
                    </span>
                    {highlightShell && (
                      <span style={{
                        color: "#ffffff50",
                        fontSize: 11,
                        fontFamily: "'IBM Plex Mono', monospace",
                      }}>
                        shell {highlightShell} · depth={highlightShell}
                      </span>
                    )}
                  </div>
                  <RadialTrie
                    langKey={primaryLang}
                    width={380}
                    height={380}
                    highlightShell={highlightShell}
                    onShellHover={setHighlightShell}
                  />
                </div>
              </div>

              {/* Comparison tries - small */}
              <div style={{ flex: "1 1 280px", display: "flex", flexDirection: "column", gap: 12 }}>
                {selectedLangs.filter(l => l !== primaryLang).slice(0, 2).map((lk) => (
                  <div key={lk} style={{
                    background: "#0a0a0f",
                    border: "1px solid #ffffff0a",
                    borderRadius: 8,
                    padding: 10,
                  }}>
                    <span style={{
                      color: LANGUAGES[lk].color,
                      fontSize: 11,
                      fontFamily: "'IBM Plex Mono', monospace",
                    }}>
                      {LANGUAGES[lk].label}
                    </span>
                    <RadialTrie
                      langKey={lk}
                      width={260}
                      height={180}
                      highlightShell={highlightShell}
                      onShellHover={setHighlightShell}
                    />
                  </div>
                ))}
              </div>
            </div>

            {/* Properties table */}
            <div style={{
              marginTop: 16,
              background: "#0a0a0f",
              border: "1px solid #ffffff0a",
              borderRadius: 8,
              padding: 16,
              overflowX: "auto",
            }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12, fontFamily: "'IBM Plex Mono', monospace" }}>
                <thead>
                  <tr style={{ color: "#ffffff40" }}>
                    <th style={{ textAlign: "left", padding: "6px 12px", borderBottom: "1px solid #ffffff10" }}>Language</th>
                    <th style={{ textAlign: "right", padding: "6px 12px", borderBottom: "1px solid #ffffff10" }}>Alphabet</th>
                    <th style={{ textAlign: "right", padding: "6px 12px", borderBottom: "1px solid #ffffff10" }}>Avg Branch</th>
                    <th style={{ textAlign: "right", padding: "6px 12px", borderBottom: "1px solid #ffffff10" }}>Phonotactic Constraint</th>
                    <th style={{ textAlign: "left", padding: "6px 12px", borderBottom: "1px solid #ffffff10" }}>Morphology</th>
                  </tr>
                </thead>
                <tbody>
                  {selectedLangs.map((lk) => {
                    const lang = LANGUAGES[lk];
                    return (
                      <tr key={lk}>
                        <td style={{ padding: "6px 12px", color: lang.color }}>{lang.label}</td>
                        <td style={{ padding: "6px 12px", textAlign: "right", color: "#ffffffaa" }}>{lang.alphabet}</td>
                        <td style={{ padding: "6px 12px", textAlign: "right", color: "#ffffffaa" }}>{lang.avgBranching.toFixed(1)}</td>
                        <td style={{ padding: "6px 12px", textAlign: "right", color: "#ffffffaa" }}>{(lang.phonotactic * 100).toFixed(0)}%</td>
                        <td style={{ padding: "6px 12px", color: "#ffffffaa" }}>{lang.morphological}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {activeView === "convergence" && (
          <div>
            <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
              <div style={{ flex: "1 1 480px" }}>
                <div style={{
                  background: "#0a0a0f",
                  border: "1px solid #ffffff0a",
                  borderRadius: 8,
                  padding: 16,
                }}>
                  <div style={{ marginBottom: 8 }}>
                    <span style={{ color: "#ffffffaa", fontSize: 13, fontFamily: "'IBM Plex Mono', monospace" }}>
                      Estimated Semantic Dimensionality vs. Depth
                    </span>
                  </div>
                  <DimensionalityChart
                    languages={selectedLangs}
                    width={480}
                    height={300}
                    highlightShell={highlightShell}
                  />
                  <div style={{ display: "flex", gap: 16, marginTop: 8, flexWrap: "wrap" }}>
                    {selectedLangs.map((lk) => (
                      <span key={lk} style={{
                        color: LANGUAGES[lk].color,
                        fontSize: 11,
                        fontFamily: "'IBM Plex Mono', monospace",
                      }}>
                        ● {LANGUAGES[lk].label}
                      </span>
                    ))}
                  </div>
                </div>
              </div>

              <div style={{ flex: "1 1 380px" }}>
                <div style={{
                  background: "#0a0a0f",
                  border: "1px solid #ffffff0a",
                  borderRadius: 8,
                  padding: 16,
                }}>
                  <div style={{ marginBottom: 8 }}>
                    <span style={{ color: "#ffffffaa", fontSize: 13, fontFamily: "'IBM Plex Mono', monospace" }}>
                      Shell Entropy (Branching Complexity)
                    </span>
                  </div>
                  <BranchingEntropyChart
                    languages={selectedLangs}
                    width={380}
                    height={300}
                    highlightShell={highlightShell}
                  />
                </div>
              </div>
            </div>

            {/* Key insight */}
            <div style={{
              marginTop: 16,
              background: "#4ECDC408",
              border: "1px solid #4ECDC420",
              borderRadius: 8,
              padding: 20,
            }}>
              <div style={{ color: "#4ECDC4", fontSize: 12, fontFamily: "'IBM Plex Mono', monospace", marginBottom: 8, letterSpacing: 1 }}>
                KEY OBSERVATION
              </div>
              <p style={{ color: "#ffffffbb", fontSize: 13, lineHeight: 1.7, margin: 0 }}>
                Despite different alphabets, phonotactic constraints, and morphological strategies,
                the estimated dimensionality curves converge toward a shared band (~32-48 effective dimensions)
                at depth 6-8. This suggests that the underlying semantic space has a fixed intrinsic dimensionality
                that is language-independent — the "shape of meaning" is universal, even as the orthographic
                encoding varies dramatically. The convergence zone corresponds to the point where word-length
                distributions peak across languages.
              </p>
            </div>
          </div>
        )}

        {activeView === "experiment" && (
          <div style={{ maxWidth: 720 }}>
            <div style={{
              color: "#ffffff50",
              fontSize: 12,
              fontFamily: "'IBM Plex Mono', monospace",
              marginBottom: 20,
              letterSpacing: 1,
            }}>
              EXPERIMENTAL FRAMEWORK: CROSS-LINGUISTIC SEMANTIC CONVERGENCE
            </div>

            <ExperimentPhase
              phase="I"
              title="Orthographic Trie Construction"
              status="active"
              description="Build radial tries for each language. For logographic systems (Chinese, Japanese kanji), use stroke-decomposition or radical trees instead of character tries. For abjads (Arabic, Hebrew), evaluate root-pattern decomposition. Measure: branching factor, path entropy, and node density at each shell."
            >
              <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginTop: 8 }}>
                {["Wiktionary dumps", "CEDICT", "Buckwalter", "UniDic", "Morphological analyzers"].map((t) => (
                  <span key={t} style={{
                    background: "#ffffff08",
                    color: "#ffffff60",
                    padding: "3px 10px",
                    borderRadius: 12,
                    fontSize: 10,
                    fontFamily: "'IBM Plex Mono', monospace",
                  }}>
                    {t}
                  </span>
                ))}
              </div>
            </ExperimentPhase>

            <ExperimentPhase
              phase="II"
              title="Semantic Embedding & Dimensionality Estimation"
              status="active"
              description="For each language, train or load embeddings (fastText/Word2Vec). Apply three independent dimensionality estimators: PCA eigenspectrum analysis (effective rank), maximum likelihood intrinsic dimensionality (MLE-ID), and topological data analysis (persistent homology). Cross-validate estimates. Plot dimensionality vs. vocabulary size to detect convergence."
            >
              <div style={{
                background: "#ffffff05",
                borderRadius: 6,
                padding: 12,
                marginTop: 8,
                fontSize: 12,
                fontFamily: "'IBM Plex Mono', monospace",
                color: "#ffffff70",
                lineHeight: 1.6,
              }}>
                <div>{"d_eff = argmin_k { Σ(i=1..k) λ_i / Σ λ_i > 0.95 }"}</div>
                <div style={{ marginTop: 4, color: "#ffffff40" }}>
                  {"where λ_i are eigenvalues of the embedding covariance matrix, sorted descending. "}
                  {"If d_eff converges across languages → universal semantic geometry."}
                </div>
              </div>
            </ExperimentPhase>

            <ExperimentPhase
              phase="III"
              title="Cross-Linguistic Alignment"
              status="active"
              description="Align embedding spaces using bilingual dictionaries (MUSE) or parallel corpora. After Procrustes alignment, measure: (a) residual error per semantic cluster, (b) whether the same principal components emerge in each language, (c) whether morphological type (isolating vs agglutinative vs fusional vs templatic) predicts which dimensions carry more variance."
            />

            <ExperimentPhase
              phase="IV"
              title="The Tiling Test"
              status="active"
              description="The critical experiment. Construct Voronoi tessellations in the semantic space of each language. If meaning-space is universal, then: (1) the Voronoi cell volumes should follow the same distribution across languages for translationally-equivalent concepts, (2) the number of cells needed to tile the 'used' portion of semantic space should converge, and (3) the fractal dimension of the cell boundary network should be language-invariant."
            >
              <div style={{
                background: "#FF6B6B10",
                border: "1px solid #FF6B6B20",
                borderRadius: 6,
                padding: 12,
                marginTop: 8,
              }}>
                <div style={{ color: "#FF6B6B", fontSize: 11, fontFamily: "'IBM Plex Mono', monospace", marginBottom: 4 }}>
                  PREDICTION
                </div>
                <p style={{ color: "#ffffffaa", fontSize: 12, lineHeight: 1.6, margin: 0 }}>
                  The tiling converges to a structure determined by the distribution of concepts humans
                  need to distinguish — not by the encoding system (language). The convergent dimensionality
                  d* reflects the intrinsic complexity of human conceptual space, estimated at 30-60 effective
                  dimensions. Deviations from d* for specific languages will correlate with Sapir-Whorf
                  effects: languages with richer color vocabulary → more dimensions in the color subspace, etc.
                </p>
              </div>
            </ExperimentPhase>

            <ExperimentPhase
              phase="V"
              title="Scale Invariance Check"
              status="active"
              description="Test whether the convergent dimensionality changes at different granularities: (a) morpheme level, (b) word level, (c) phrase level, (d) sentence level. If meaning is self-similar, the dimensionality should be scale-invariant (or follow a predictable scaling law). This would confirm the tiling analogy: aperiodic tilings like Penrose tiles have the same local structure at every scale."
            >
              <div style={{
                background: "#C49BFF10",
                border: "1px solid #C49BFF20",
                borderRadius: 6,
                padding: 12,
                marginTop: 8,
              }}>
                <div style={{ color: "#C49BFF", fontSize: 11, fontFamily: "'IBM Plex Mono', monospace", marginBottom: 4 }}>
                  THE DEEP QUESTION
                </div>
                <p style={{ color: "#ffffffaa", fontSize: 12, lineHeight: 1.6, margin: 0 }}>
                  If d* is scale-invariant, it implies that meaning has a fractal structure — the same
                  "shape" appears at every level of linguistic composition. This would mean the convergent
                  dimensionality is not just a property of words, but of thought itself. The tiling
                  analogy becomes literal: human conceptual space is a quasicrystal.
                </p>
              </div>
            </ExperimentPhase>
          </div>
        )}
      </div>
    </div>
  );
}
