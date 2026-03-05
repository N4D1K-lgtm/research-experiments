// ═══════════════════════════════════════════════════════════════════════════
// The Shape of Meaning — Interactive Essay Engine
// Uses real IPA trie data from the project pipeline
// ═══════════════════════════════════════════════════════════════════════════

// ── Types ─────────────────────────────────────────────────────────────────

interface CompactNode {
  id: number;
  ph: string;      // phoneme
  d: number;       // depth
  pid: number | null;
  role: string;    // o/n/c/m
  cnt: number;     // totalCount
  term: boolean;
  x: number; y: number; z: number;
  tp?: Record<string, number>;  // top transition probs
  w?: string[];    // sample words
}

interface EssayData {
  nodes: CompactNode[];
  meta: {
    totalNodes: number;
    totalEdges: number;
    totalWords: number;
    totalTerminals: number;
    phonemeInventory: string[];
    onsetInventory: string[];
    codaInventory: string[];
    maxDepth: number;
    motifs: { seq: string; count: number }[];
  };
  transitionMatrix: Record<string, Record<string, number>>;
}

interface CrossLingStats {
  [lang: string]: {
    depth: number;
    nodes: number;
    terminals: number;
    avgBranch: number;
    avgEntropy: number;
    maxEntropy: number;
  }[];
}

// ── Globals ───────────────────────────────────────────────────────────────

let essayData: EssayData | null = null;
let crossLingStats: CrossLingStats | null = null;

// ── Utilities ─────────────────────────────────────────────────────────────

function dpr(): number {
  return Math.min(window.devicePixelRatio || 1, 2);
}

function initCanvas(id: string): { canvas: HTMLCanvasElement; ctx: CanvasRenderingContext2D; w: number; h: number } | null {
  const canvas = document.getElementById(id) as HTMLCanvasElement | null;
  if (!canvas) return null;
  const rect = canvas.getBoundingClientRect();
  const w = rect.width || canvas.clientWidth || 760;
  const h = canvas.height;
  const d = dpr();
  canvas.width = w * d;
  canvas.height = h * d;
  canvas.style.width = w + "px";
  canvas.style.height = h + "px";
  const ctx = canvas.getContext("2d")!;
  ctx.scale(d, d);
  return { canvas, ctx, w, h };
}

function seedRandom(seed: number): () => number {
  let s = seed;
  return () => { s = (s * 16807 + 0) % 2147483647; return s / 2147483647; };
}

function hexAlpha(hex: string, alpha: number): string {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r},${g},${b},${Math.max(0, Math.min(1, alpha))})`;
}

function roundRect(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number): void {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - r);
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
}

const COLORS = {
  bg: "#0a0a12",
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
  textDim: "#6a6a7a",
  textFaint: "#3a3a4a",
  border: "rgba(255,255,255,0.06)",
};

const ROLE_COLORS: Record<string, string> = { o: COLORS.onset, n: COLORS.nucleus, c: COLORS.coda, m: COLORS.mixed };
const ROLE_NAMES: Record<string, string> = { o: "onset", n: "nucleus", c: "coda", m: "mixed" };

const LANG_CONFIG: Record<string, { label: string; color: string; family: string; type: string }> = {
  en_US: { label: "English", color: COLORS.accent, family: "Germanic", type: "Concatenative" },
  fr_FR: { label: "French", color: COLORS.red, family: "Romance", type: "Concatenative" },
  es_ES: { label: "Spanish", color: COLORS.yellow, family: "Romance", type: "Concatenative" },
  de: { label: "German", color: COLORS.purple, family: "Germanic", type: "Concatenative" },
  nl: { label: "Dutch", color: COLORS.blue, family: "Germanic", type: "Concatenative" },
};

// ── Data Loading ──────────────────────────────────────────────────────────

async function loadData(): Promise<void> {
  try {
    const [essayResp, crossResp] = await Promise.all([
      fetch("/essay_data.json"),
      fetch("/cross_linguistic_stats.json"),
    ]);
    if (essayResp.ok) essayData = await essayResp.json();
    if (crossResp.ok) crossLingStats = await crossResp.json();
  } catch {
    console.warn("Data files not found, using embedded fallbacks");
  }
}

// ── Scroll & Navigation ──────────────────────────────────────────────────

function initScrollEngine(): void {
  const progress = document.getElementById("progress")!;
  const navDots = document.querySelectorAll<HTMLButtonElement>(".nav-dot");
  const sections = document.querySelectorAll<HTMLElement>("section");
  const reveals = document.querySelectorAll<HTMLElement>(".reveal");

  window.addEventListener("scroll", () => {
    const scrollTop = window.scrollY;
    const docHeight = document.documentElement.scrollHeight - window.innerHeight;
    const pct = docHeight > 0 ? (scrollTop / docHeight) * 100 : 0;
    progress.style.width = pct + "%";
  });

  navDots.forEach((dot) => {
    dot.addEventListener("click", () => {
      const target = document.getElementById(dot.dataset.section || "");
      target?.scrollIntoView({ behavior: "smooth" });
    });
  });

  const navObserver = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          navDots.forEach((d) => d.classList.remove("active"));
          const dot = document.querySelector(`.nav-dot[data-section="${entry.target.id}"]`);
          dot?.classList.add("active");
        }
      });
    },
    { threshold: 0.3 },
  );
  sections.forEach((s) => navObserver.observe(s));

  const revealObserver = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          (entry.target as HTMLElement).classList.add("visible");
        }
      });
    },
    { threshold: 0.1 },
  );
  reveals.forEach((r) => revealObserver.observe(r));
}

// ── Hero Background ──────────────────────────────────────────────────────

function initHeroBackground(): void {
  const canvas = document.getElementById("hero-bg") as HTMLCanvasElement | null;
  if (!canvas) return;
  const w = window.innerWidth;
  const h = window.innerHeight;
  const d = dpr();
  canvas.width = w * d;
  canvas.height = h * d;
  canvas.style.width = w + "px";
  canvas.style.height = h + "px";
  const ctx = canvas.getContext("2d")!;
  ctx.scale(d, d);

  const rng = seedRandom(42);
  const cx = w / 2;
  const cy = h / 2;

  ctx.fillStyle = COLORS.bg;
  ctx.fillRect(0, 0, w, h);

  // Draw real trie structure if we have data
  if (essayData) {
    const nodes = essayData.nodes.filter(n => n.d <= 4);
    const maxR = Math.min(w, h) * 0.42;

    // Shells
    for (let d = 1; d <= 6; d++) {
      const radius = (d / 6) * maxR;
      ctx.beginPath();
      ctx.arc(cx, cy, radius, 0, Math.PI * 2);
      ctx.strokeStyle = `rgba(78,205,196,${0.04 - d * 0.005})`;
      ctx.lineWidth = 0.5;
      ctx.stroke();
    }

    // Sample nodes with angular distribution based on real data
    for (const n of nodes) {
      if (n.d === 0) continue;
      const radius = (n.d / 6) * maxR;
      // Use node id to seed angle for consistency
      const angle = ((n.id * 2.3999632) % (Math.PI * 2));
      const x = cx + Math.cos(angle) * radius;
      const y = cy + Math.sin(angle) * radius;
      const size = 0.3 + Math.log10(n.cnt + 1) * 0.3;
      const alpha = 0.02 + Math.log10(n.cnt + 1) * 0.01;
      const color = ROLE_COLORS[n.role] || COLORS.accent;
      ctx.beginPath();
      ctx.arc(x, y, Math.min(size, 2), 0, Math.PI * 2);
      ctx.fillStyle = hexAlpha(color, alpha);
      ctx.fill();
    }
  } else {
    // Fallback: generative background
    for (let d = 1; d <= 10; d++) {
      const radius = d * Math.min(w, h) / 22;
      ctx.beginPath();
      ctx.arc(cx, cy, radius, 0, Math.PI * 2);
      ctx.strokeStyle = `rgba(78,205,196,${0.04 - d * 0.003})`;
      ctx.lineWidth = 0.5;
      ctx.stroke();
      const count = Math.floor(12 * Math.pow(1.3, d));
      for (let i = 0; i < count; i++) {
        const angle = rng() * Math.PI * 2;
        const x = cx + Math.cos(angle) * radius;
        const y = cy + Math.sin(angle) * radius;
        ctx.beginPath();
        ctx.arc(x, y, 0.5 + rng() * 1.5, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(78,205,196,${0.03 + rng() * 0.08})`;
        ctx.fill();
      }
    }
  }

  // Center glow
  const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, 60);
  grad.addColorStop(0, "rgba(78,205,196,0.08)");
  grad.addColorStop(1, "rgba(78,205,196,0)");
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, w, h);
}

// ── 1. IPA Feature Space ─────────────────────────────────────────────────
// Uses the REAL phoneme inventory from our trie data

interface PhonemeInfo {
  symbol: string;
  voiced: boolean;
  place: string;
  manner: string;
  inInventory: boolean;  // whether this appears in our English data
  count?: number;
  x: number;
  y: number;
}

// Full IPA consonant classification (articulatory features)
const IPA_FEATURES: Record<string, { voiced: boolean; place: string; manner: string }> = {
  "p": { voiced: false, place: "Bilabial", manner: "Plosive" },
  "b": { voiced: true, place: "Bilabial", manner: "Plosive" },
  "t": { voiced: false, place: "Alveolar", manner: "Plosive" },
  "d": { voiced: true, place: "Alveolar", manner: "Plosive" },
  "k": { voiced: false, place: "Velar", manner: "Plosive" },
  "\u0261": { voiced: true, place: "Velar", manner: "Plosive" },  // ɡ
  "m": { voiced: true, place: "Bilabial", manner: "Nasal" },
  "n": { voiced: true, place: "Alveolar", manner: "Nasal" },
  "\u014B": { voiced: true, place: "Velar", manner: "Nasal" },    // ŋ
  "f": { voiced: false, place: "Labiodental", manner: "Fricative" },
  "v": { voiced: true, place: "Labiodental", manner: "Fricative" },
  "\u03B8": { voiced: false, place: "Dental", manner: "Fricative" },  // θ
  "\u00F0": { voiced: true, place: "Dental", manner: "Fricative" },   // ð
  "s": { voiced: false, place: "Alveolar", manner: "Fricative" },
  "z": { voiced: true, place: "Alveolar", manner: "Fricative" },
  "\u0283": { voiced: false, place: "Post-alv.", manner: "Fricative" },  // ʃ
  "\u0292": { voiced: true, place: "Post-alv.", manner: "Fricative" },   // ʒ
  "h": { voiced: false, place: "Glottal", manner: "Fricative" },
  "j": { voiced: true, place: "Palatal", manner: "Approximant" },
  "w": { voiced: true, place: "Labio-velar", manner: "Approximant" },
  "\u0279": { voiced: true, place: "Alveolar", manner: "Approximant" },  // ɹ
  "l": { voiced: true, place: "Alveolar", manner: "Lateral" },
};

// IPA vowel chart positions: [height, backness]
const IPA_VOWELS: Record<string, { height: string; backness: string; rounded: boolean; label: string }> = {
  "i": { height: "Close", backness: "Front", rounded: false, label: "i" },
  "u": { height: "Close", backness: "Back", rounded: true, label: "u" },
  "\u026A": { height: "Near-close", backness: "Front", rounded: false, label: "\u026A" },  // ɪ
  "\u028A": { height: "Near-close", backness: "Back", rounded: true, label: "\u028A" },     // ʊ
  "e": { height: "Close-mid", backness: "Front", rounded: false, label: "e" },
  "o": { height: "Close-mid", backness: "Back", rounded: true, label: "o" },
  "\u0259": { height: "Mid", backness: "Central", rounded: false, label: "\u0259" },        // ə
  "\u025B": { height: "Open-mid", backness: "Front", rounded: false, label: "\u025B" },     // ɛ
  "\u0254": { height: "Open-mid", backness: "Back", rounded: true, label: "\u0254" },       // ɔ
  "\u00E6": { height: "Near-open", backness: "Front", rounded: false, label: "\u00E6" },    // æ
  "a": { height: "Open", backness: "Front", rounded: false, label: "a" },
  "\u0251": { height: "Open", backness: "Back", rounded: false, label: "\u0251" },          // ɑ
  "\u025D": { height: "Mid", backness: "Central", rounded: false, label: "\u025D" },        // ɝ (rhotacized schwa)
};

function initIPAViz(): void {
  const r = initCanvas("viz-ipa");
  if (!r) return;
  const { canvas, ctx, w, h } = r;

  // Build phoneme list from real inventory
  const inventory = new Set(essayData?.meta.phonemeInventory || []);

  // Count occurrences from depth-1 nodes (initial position gives frequency)
  const phonemeCounts: Record<string, number> = {};
  if (essayData) {
    for (const n of essayData.nodes) {
      if (n.d === 1) {
        phonemeCounts[n.ph] = n.cnt;
      }
    }
  }

  const places = ["Bilabial", "Labiodental", "Dental", "Alveolar", "Post-alv.", "Palatal", "Labio-velar", "Velar", "Glottal"];
  const manners = ["Plosive", "Nasal", "Fricative", "Approximant", "Lateral"];

  const phonemes: PhonemeInfo[] = [];
  for (const [sym, feat] of Object.entries(IPA_FEATURES)) {
    phonemes.push({
      symbol: sym,
      voiced: feat.voiced,
      place: feat.place,
      manner: feat.manner,
      inInventory: inventory.has(sym),
      count: phonemeCounts[sym],
      x: 0, y: 0,
    });
  }

  const pad = { top: 44, left: 90, right: 20, bottom: 60 };
  const cellW = (w - pad.left - pad.right) / places.length;
  const cellH = (h - pad.top - pad.bottom) / manners.length;

  phonemes.forEach((p) => {
    const pi = places.indexOf(p.place);
    const mi = manners.indexOf(p.manner);
    const voiceOffset = p.voiced ? cellW * 0.28 : -cellW * 0.28;
    p.x = pad.left + (pi + 0.5) * cellW + voiceOffset;
    p.y = pad.top + (mi + 0.5) * cellH;
  });

  let hovered: PhonemeInfo | null = null;

  function draw(): void {
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = COLORS.bg;
    ctx.fillRect(0, 0, w, h);

    // Grid
    ctx.strokeStyle = COLORS.border;
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= places.length; i++) {
      const x = pad.left + i * cellW;
      ctx.beginPath(); ctx.moveTo(x, pad.top); ctx.lineTo(x, h - pad.bottom); ctx.stroke();
    }
    for (let i = 0; i <= manners.length; i++) {
      const y = pad.top + i * cellH;
      ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(w - pad.right, y); ctx.stroke();
    }

    // Column/row labels
    ctx.font = "9px 'IBM Plex Mono', monospace";
    ctx.fillStyle = COLORS.textDim;
    ctx.textAlign = "center";
    places.forEach((p, i) => ctx.fillText(p, pad.left + (i + 0.5) * cellW, pad.top - 10));
    ctx.textAlign = "right";
    manners.forEach((m, i) => ctx.fillText(m, pad.left - 8, pad.top + (i + 0.5) * cellH + 4));

    // Phonemes
    phonemes.forEach((p) => {
      const isHov = hovered === p;
      const inInv = p.inInventory;
      const size = isHov ? 18 : 14;

      // Frequency-based background
      if (inInv && p.count) {
        const freqAlpha = Math.min(0.15, Math.log10(p.count + 1) * 0.03);
        ctx.beginPath();
        ctx.arc(p.x, p.y, 14, 0, Math.PI * 2);
        ctx.fillStyle = hexAlpha(p.voiced ? COLORS.accent : COLORS.purple, freqAlpha);
        ctx.fill();
      }

      if (isHov) {
        ctx.beginPath();
        ctx.arc(p.x, p.y, 17, 0, Math.PI * 2);
        ctx.fillStyle = p.voiced ? "rgba(78,205,196,0.2)" : "rgba(196,155,255,0.2)";
        ctx.fill();
      }

      ctx.font = `${isHov ? 600 : 400} ${size}px 'IBM Plex Mono', monospace`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";

      const alpha = inInv ? (isHov ? 1 : 0.85) : 0.25;
      ctx.fillStyle = p.voiced
        ? hexAlpha(COLORS.accent, alpha)
        : hexAlpha(COLORS.purple, alpha);
      ctx.fillText(p.symbol, p.x, p.y);
    });

    // Hover tooltip
    if (hovered) {
      const infoX = w - 210;
      const infoY = 14;
      ctx.fillStyle = "rgba(10,10,18,0.92)";
      ctx.strokeStyle = "rgba(255,255,255,0.1)";
      ctx.lineWidth = 1;
      roundRect(ctx, infoX, infoY, 196, hovered.count ? 96 : 76, 8);
      ctx.fill(); ctx.stroke();

      ctx.font = "22px 'IBM Plex Mono', monospace";
      ctx.fillStyle = hovered.voiced ? COLORS.accent : COLORS.purple;
      ctx.textAlign = "left"; ctx.textBaseline = "top";
      ctx.fillText(`/${hovered.symbol}/`, infoX + 12, infoY + 10);

      ctx.font = "10px 'IBM Plex Mono', monospace";
      ctx.fillStyle = COLORS.textDim;
      ctx.fillText(`[${hovered.voiced ? "+voice" : "-voice"}] ${hovered.place.toLowerCase()}`, infoX + 12, infoY + 40);
      ctx.fillText(hovered.manner.toLowerCase(), infoX + 12, infoY + 56);

      if (hovered.count) {
        ctx.fillStyle = hovered.inInventory ? "rgba(255,255,255,0.5)" : COLORS.textFaint;
        ctx.fillText(`${hovered.count.toLocaleString()} occurrences in English trie`, infoX + 12, infoY + 76);
      }
    }

    // Legend
    ctx.font = "10px 'IBM Plex Mono', monospace";
    ctx.textAlign = "left"; ctx.textBaseline = "alphabetic";
    ctx.fillStyle = COLORS.purple;
    ctx.fillText("\u25CF voiceless", pad.left, h - 32);
    ctx.fillStyle = COLORS.accent;
    ctx.fillText("\u25CF voiced", pad.left + 100, h - 32);
    ctx.fillStyle = COLORS.textFaint;
    ctx.fillText("brightness \u221D frequency in corpus", pad.left + 190, h - 32);
    if (inventory.size > 0) {
      ctx.fillText(`English inventory: ${inventory.size} consonants shown (${essayData?.meta.phonemeInventory.length} total phonemes)`, pad.left, h - 14);
    }
  }

  canvas.addEventListener("mousemove", (e) => {
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    hovered = null;
    for (const p of phonemes) {
      if (Math.abs(mx - p.x) < 14 && Math.abs(my - p.y) < 14) { hovered = p; break; }
    }
    canvas.style.cursor = hovered ? "pointer" : "default";
    draw();
  });
  canvas.addEventListener("mouseleave", () => { hovered = null; draw(); });
  draw();
}

// ── 1b. Tokenizer (uses real tokenization logic from pipeline) ────────────

function initTokenizer(): void {
  const input = document.getElementById("tokenizer-input") as HTMLInputElement;
  const output = document.getElementById("tokenizer-output")!;
  const chips = document.querySelectorAll<HTMLButtonElement>(".word-chip:not(.trie-word)");

  // Exact same tokenization rules as scripts/build_trie.py
  const COMBINING_START = 0x0300;
  const COMBINING_END = 0x036F;
  const IPA_DIACRITICS = new Set([
    "\u0329", "\u032F", "\u0325", "\u031F", "\u0320", "\u0303", "\u02D0",
    "\u02B0", "\u02B7", "\u02B2", "\u0361", "\u035C",
  ]);
  const SUPRASEGMENTALS = new Set(["\u02C8", "\u02CC", ".", "\u203F", "|", "\u2016"]);

  // Real vowel set from IPA
  const VOWELS = new Set(["a", "e", "i", "o", "u", "\u0259", "\u025B", "\u0254",
    "\u026A", "\u028A", "\u00E6", "\u0251", "\u0252", "\u028C", "\u025C",
    "\u0258", "\u0275", "\u0264", "\u026F", "\u0268", "\u0289", "y",
    "\u00F8", "\u0153", "\u0276", "\u0250", "\u025E", "\u0269", "\u025D"]);

  function tokenize(ipa: string): string[] {
    const clean = ipa.replace(/[/\[\]]/g, "");
    const tokens: string[] = [];
    let i = 0;
    while (i < clean.length) {
      const ch = clean[i];
      if (SUPRASEGMENTALS.has(ch) || ch === " ") { i++; continue; }
      let token = ch;
      i++;
      // Absorb combining diacritics
      while (i < clean.length) {
        const next = clean[i];
        const cp = next.codePointAt(0) || 0;
        if ((cp >= COMBINING_START && cp <= COMBINING_END) || IPA_DIACRITICS.has(next)) {
          token += next; i++;
        } else { break; }
      }
      if (token.trim()) tokens.push(token);
    }
    return tokens;
  }

  function classify(token: string): string {
    const base = token[0];
    if (VOWELS.has(base)) return "nucleus";
    // Check real inventory if available
    if (essayData) {
      if (essayData.meta.onsetInventory.includes(base) && !essayData.meta.codaInventory.includes(base)) return "onset";
      if (essayData.meta.codaInventory.includes(base) && !essayData.meta.onsetInventory.includes(base)) return "coda";
      if (essayData.meta.onsetInventory.includes(base)) return "onset";
    }
    return "onset";
  }

  function render(ipa: string): void {
    const tokens = tokenize(ipa);
    output.innerHTML = "";
    if (tokens.length === 0) {
      output.innerHTML = `<span style="color: ${COLORS.textFaint}; font-size: 12px;">Type IPA to see tokenization</span>`;
      return;
    }
    tokens.forEach((t, i) => {
      const cls = classify(t);
      const span = document.createElement("span");
      span.className = `token ${cls}`;
      span.textContent = t;
      span.title = `${ROLE_NAMES[cls[0]] || cls} \u2014 /${t}/`;
      output.appendChild(span);
      if (i < tokens.length - 1) {
        const arrow = document.createElement("span");
        arrow.className = "token-arrow";
        arrow.textContent = "\u2192";
        output.appendChild(arrow);
      }
    });
  }

  input.addEventListener("input", () => render(input.value));
  chips.forEach((chip) => {
    chip.addEventListener("click", () => {
      input.value = chip.dataset.ipa || "";
      render(input.value);
      chips.forEach((c) => c.classList.remove("active"));
      chip.classList.add("active");
    });
  });
  render("\u02C8k\u00E6t");
}

// ── 2. Trie Builder ──────────────────────────────────────────────────────

interface TrieNode {
  char: string;
  children: Map<string, TrieNode>;
  isWord: boolean;
  words: string[];
  depth: number;
  x: number; y: number;
}

function initTrieBuilder(): void {
  const r = initCanvas("viz-trie");
  if (!r) return;
  const { canvas, ctx, w, h } = r;

  const root: TrieNode = { char: "\u2022", children: new Map(), isWord: false, words: [], depth: 0, x: 0, y: 0 };
  const activeWords = new Set(["cat"]);

  function insertWord(word: string): void {
    let node = root;
    for (let i = 0; i < word.length; i++) {
      const ch = word[i];
      if (!node.children.has(ch)) {
        node.children.set(ch, { char: ch, children: new Map(), isWord: false, words: [], depth: i + 1, x: 0, y: 0 });
      }
      node = node.children.get(ch)!;
    }
    node.isWord = true;
    if (!node.words.includes(word)) node.words.push(word);
  }

  function countLeaves(node: TrieNode): number {
    const children = Array.from(node.children.values());
    if (children.length === 0) return 1;
    return children.reduce((sum, c) => sum + countLeaves(c), 0);
  }

  function layoutTrie(node: TrieNode, xMin: number, xMax: number, y: number, yStep: number): void {
    node.x = (xMin + xMax) / 2;
    node.y = y;
    const children = Array.from(node.children.values());
    if (children.length === 0) return;
    const totalLeaves = children.reduce((sum, c) => sum + countLeaves(c), 0);
    let xCursor = xMin;
    children.forEach((child) => {
      const share = countLeaves(child) / totalLeaves;
      layoutTrie(child, xCursor, xCursor + (xMax - xMin) * share, y + yStep, yStep);
      xCursor += (xMax - xMin) * share;
    });
  }

  function drawNode(node: TrieNode): void {
    node.children.forEach((child) => {
      ctx.beginPath();
      ctx.moveTo(node.x, node.y);
      // Curved edge
      const midY = (node.y + child.y) / 2;
      ctx.quadraticCurveTo(node.x, midY, child.x, child.y);
      ctx.strokeStyle = child.isWord ? "rgba(78,205,196,0.4)" : "rgba(255,255,255,0.1)";
      ctx.lineWidth = child.isWord ? 1.5 : 0.8;
      ctx.stroke();
      drawNode(child);
    });

    const radius = node.depth === 0 ? 8 : node.isWord ? 6 : 4;
    ctx.beginPath();
    ctx.arc(node.x, node.y, radius, 0, Math.PI * 2);
    ctx.fillStyle = node.depth === 0 ? COLORS.accent : node.isWord ? "rgba(78,205,196,0.9)" : "rgba(255,255,255,0.15)";
    ctx.fill();

    ctx.font = `${node.depth === 0 ? 12 : node.isWord ? 14 : 12}px 'IBM Plex Mono', monospace`;
    ctx.fillStyle = node.isWord ? "#e0e0ec" : "rgba(255,255,255,0.5)";
    ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillText(node.char, node.x, node.y - radius - 10);

    if (node.isWord && node.words.length > 0) {
      ctx.font = "10px 'IBM Plex Mono', monospace";
      ctx.fillStyle = "rgba(78,205,196,0.5)";
      ctx.fillText(node.words.join(", "), node.x, node.y + radius + 12);
    }
  }

  function draw(): void {
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = COLORS.bg;
    ctx.fillRect(0, 0, w, h);

    root.children.clear(); root.isWord = false; root.words = [];
    activeWords.forEach((word) => insertWord(word));
    layoutTrie(root, 40, w - 40, 40, 70);
    drawNode(root);

    // Stats
    let nc = 0;
    function count(n: TrieNode): void { nc++; n.children.forEach((c) => count(c)); }
    count(root);
    const md = (function maxD(n: TrieNode): number { let m = n.depth; n.children.forEach(c => { m = Math.max(m, maxD(c)); }); return m; })(root);
    ctx.font = "10px 'IBM Plex Mono', monospace";
    ctx.fillStyle = COLORS.textDim;
    ctx.textAlign = "left";
    ctx.fillText(`${activeWords.size} words \u00B7 ${nc} nodes \u00B7 max depth ${md} \u00B7 prefix sharing saves ${activeWords.size * 3 - nc + 1} nodes`, 12, h - 10);
  }

  document.querySelectorAll<HTMLButtonElement>(".trie-word").forEach((chip) => {
    chip.addEventListener("click", () => {
      const word = chip.dataset.word || "";
      if (activeWords.has(word)) { activeWords.delete(word); chip.classList.remove("active"); }
      else { activeWords.add(word); chip.classList.add("active"); }
      draw();
    });
  });
  draw();
}

// ── 3. Radial Trie (REAL DATA) ───────────────────────────────────────────

function initRadialTrie(): void {
  const r = initCanvas("viz-radial");
  if (!r) return;
  const { canvas, ctx, w, h } = r;

  if (!essayData) {
    ctx.fillStyle = COLORS.bg; ctx.fillRect(0, 0, w, h);
    ctx.font = "12px 'IBM Plex Mono', monospace";
    ctx.fillStyle = COLORS.textDim; ctx.textAlign = "center";
    ctx.fillText("Trie data not loaded \u2014 run the build pipeline first", w / 2, h / 2);
    return;
  }

  const nodes = essayData.nodes;
  const maxDepth = 6; // we only have depth <= 6 in compact data
  let hoveredShell: number | null = null;

  // Pre-compute angular positions using golden angle
  const phi = (1 + Math.sqrt(5)) / 2;
  const goldenAngle = 2 * Math.PI / (phi * phi);

  // Group nodes by depth and assign angles
  const depthGroups: Map<number, CompactNode[]> = new Map();
  for (const n of nodes) {
    if (!depthGroups.has(n.d)) depthGroups.set(n.d, []);
    depthGroups.get(n.d)!.push(n);
  }

  // Assign 2D radial position
  const nodePositions = new Map<number, { x: number; y: number }>();
  for (const [depth, group] of depthGroups) {
    // Sort by parent for angular coherence
    group.sort((a, b) => (a.pid || 0) - (b.pid || 0));
    group.forEach((n, i) => {
      const angle = i * goldenAngle;
      nodePositions.set(n.id, { x: Math.cos(angle), y: Math.sin(angle) });
    });
  }

  function draw(): void {
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = COLORS.bg;
    ctx.fillRect(0, 0, w, h);

    const cx = w / 2;
    const cy = h / 2;
    const maxR = Math.min(w, h) / 2 - 30;
    const shellR = (d: number) => (d / maxDepth) * maxR;

    // Shell rings
    for (let d = 1; d <= maxDepth; d++) {
      ctx.beginPath();
      ctx.arc(cx, cy, shellR(d), 0, Math.PI * 2);
      const isHl = hoveredShell === d;
      ctx.strokeStyle = isHl ? "rgba(78,205,196,0.25)" : "rgba(255,255,255,0.03)";
      ctx.lineWidth = isHl ? 1.5 : 0.5;
      ctx.stroke();

      if (isHl) {
        ctx.fillStyle = "rgba(78,205,196,0.03)";
        ctx.beginPath();
        ctx.arc(cx, cy, shellR(d), 0, Math.PI * 2);
        ctx.arc(cx, cy, d > 1 ? shellR(d - 1) : 0, 0, Math.PI * 2, true);
        ctx.fill();
      }
    }

    // Draw only a sample of edges for performance (high-frequency paths)
    const maxCntAtD1 = Math.max(...nodes.filter(n => n.d === 1).map(n => n.cnt));
    for (const n of nodes) {
      if (n.d === 0 || n.d > maxDepth || !n.pid) continue;
      // Only draw edges for nodes with significant frequency
      if (n.d > 2 && n.cnt < maxCntAtD1 * 0.002) continue;

      const pos = nodePositions.get(n.id);
      const ppos = nodePositions.get(n.pid);
      if (!pos || !ppos) continue;

      const parentNode = nodes.find(p => p.id === n.pid);
      if (!parentNode) continue;

      const x1 = cx + ppos.x * shellR(parentNode.d);
      const y1 = cy + ppos.y * shellR(parentNode.d);
      const x2 = cx + pos.x * shellR(n.d);
      const y2 = cy + pos.y * shellR(n.d);

      const freqRatio = n.cnt / maxCntAtD1;
      const alpha = Math.min(0.3, 0.01 + freqRatio * 0.25);

      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.strokeStyle = `rgba(255,255,255,${alpha})`;
      ctx.lineWidth = 0.3 + freqRatio * 1.5;
      ctx.stroke();
    }

    // Draw nodes
    for (const n of nodes) {
      if (n.d === 0 || n.d > maxDepth) continue;
      const pos = nodePositions.get(n.id);
      if (!pos) continue;
      const x = cx + pos.x * shellR(n.d);
      const y = cy + pos.y * shellR(n.d);

      const brightness = Math.min(1, 0.15 + Math.log10(n.cnt + 1) / Math.log10(maxCntAtD1 + 1));
      const nodeR = n.term ? 1.5 + brightness * 2 : 0.4 + brightness * 0.6;
      const color = ROLE_COLORS[n.role] || COLORS.mixed;

      ctx.beginPath();
      ctx.arc(x, y, nodeR, 0, Math.PI * 2);
      ctx.fillStyle = hexAlpha(color, n.term ? brightness * 0.85 : brightness * 0.35);
      ctx.fill();
    }

    // Center
    ctx.beginPath();
    ctx.arc(cx, cy, 5, 0, Math.PI * 2);
    ctx.fillStyle = COLORS.accent;
    ctx.fill();

    // Shell stats on hover
    if (hoveredShell !== null) {
      const group = depthGroups.get(hoveredShell);
      if (group) {
        const terminals = group.filter(n => n.term).length;
        const prevGroup = depthGroups.get(hoveredShell - 1);
        const bf = prevGroup ? (group.length / prevGroup.length).toFixed(2) : "n/a";
        ctx.font = "11px 'IBM Plex Mono', monospace";
        ctx.fillStyle = "rgba(255,255,255,0.7)";
        ctx.textAlign = "left";
        ctx.fillText(
          `Shell ${hoveredShell}: ${group.length.toLocaleString()} nodes \u00B7 ${terminals.toLocaleString()} terminals \u00B7 branching factor ${bf}`,
          12, 20,
        );
      }
    }

    // Global stats
    ctx.font = "10px 'IBM Plex Mono', monospace";
    ctx.fillStyle = COLORS.textFaint;
    ctx.textAlign = "left";
    ctx.fillText(
      `${essayData!.meta.totalNodes.toLocaleString()} nodes \u00B7 ${essayData!.meta.totalTerminals.toLocaleString()} terminals \u00B7 ${essayData!.meta.phonemeInventory.length} phonemes \u00B7 ${essayData!.meta.totalWords.toLocaleString()} words`,
      12, h - 10,
    );
  }

  canvas.addEventListener("mousemove", (e) => {
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left - w / 2;
    const my = e.clientY - rect.top - h / 2;
    const dist = Math.sqrt(mx * mx + my * my);
    const maxR = Math.min(w, h) / 2 - 30;
    const shell = Math.round((dist / maxR) * maxDepth);
    hoveredShell = shell >= 1 && shell <= maxDepth ? shell : null;
    draw();
  });
  canvas.addEventListener("mouseleave", () => { hoveredShell = null; draw(); });
  draw();
}

// ── 4. Entropy Chart (REAL CROSS-LINGUISTIC DATA) ─────────────────────────

function initEntropyChart(): void {
  const r = initCanvas("viz-entropy");
  if (!r) return;
  const { ctx, w, h } = r;

  const pad = { top: 30, right: 60, bottom: 40, left: 55 };
  const pw = w - pad.left - pad.right;
  const ph = h - pad.top - pad.bottom;

  ctx.fillStyle = COLORS.bg;
  ctx.fillRect(0, 0, w, h);

  if (!crossLingStats) {
    ctx.font = "12px 'IBM Plex Mono', monospace";
    ctx.fillStyle = COLORS.textDim; ctx.textAlign = "center";
    ctx.fillText("Cross-linguistic data not loaded", w / 2, h / 2);
    return;
  }

  // Find global maxima
  let maxBranch = 0;
  let maxEntropy = 0;
  let maxDepth = 0;
  for (const [, stats] of Object.entries(crossLingStats)) {
    for (const s of stats) {
      maxBranch = Math.max(maxBranch, s.avgBranch);
      maxEntropy = Math.max(maxEntropy, s.avgEntropy);
      maxDepth = Math.max(maxDepth, s.depth);
    }
  }
  maxBranch = Math.ceil(maxBranch + 1);
  maxEntropy = Math.ceil(maxEntropy * 2) / 2 + 0.5;
  maxDepth = Math.min(maxDepth, 12);

  // Grid
  ctx.strokeStyle = "rgba(255,255,255,0.04)";
  ctx.lineWidth = 0.5;
  for (let i = 0; i <= 5; i++) {
    const y = pad.top + (i / 5) * ph;
    ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(pad.left + pw, y); ctx.stroke();
  }

  // Draw branching (solid) and entropy (dashed) for each language
  for (const [lang, stats] of Object.entries(crossLingStats)) {
    const conf = LANG_CONFIG[lang];
    if (!conf) continue;

    // Branching factor (solid)
    ctx.beginPath();
    let first = true;
    for (const s of stats) {
      if (s.depth > maxDepth) break;
      const x = pad.left + (s.depth / maxDepth) * pw;
      const y = pad.top + (1 - s.avgBranch / maxBranch) * ph;
      first ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      first = false;
    }
    ctx.strokeStyle = hexAlpha(conf.color, 0.75);
    ctx.lineWidth = 2;
    ctx.setLineDash([]);
    ctx.stroke();

    // Entropy (dashed)
    ctx.beginPath();
    first = true;
    for (const s of stats) {
      if (s.depth > maxDepth) break;
      const x = pad.left + (s.depth / maxDepth) * pw;
      const y = pad.top + (1 - s.avgEntropy / maxEntropy) * ph;
      first ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      first = false;
    }
    ctx.strokeStyle = hexAlpha(conf.color, 0.35);
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 4]);
    ctx.stroke();
  }
  ctx.setLineDash([]);

  // X axis
  ctx.font = "10px 'IBM Plex Mono', monospace";
  ctx.fillStyle = COLORS.textDim;
  ctx.textAlign = "center";
  for (let d = 0; d <= maxDepth; d += 2) {
    ctx.fillText(d.toString(), pad.left + (d / maxDepth) * pw, pad.top + ph + 20);
  }
  ctx.fillText("phoneme depth", pad.left + pw / 2, pad.top + ph + 36);

  // Left Y (branching)
  ctx.textAlign = "right";
  for (let i = 0; i <= 5; i++) {
    const val = (i / 5) * maxBranch;
    ctx.fillText(val.toFixed(0), pad.left - 8, pad.top + (1 - val / maxBranch) * ph + 4);
  }
  ctx.save();
  ctx.translate(12, pad.top + ph / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = "center";
  ctx.fillText("avg branching factor (solid)", 0, 0);
  ctx.restore();

  // Right Y (entropy)
  ctx.textAlign = "left";
  for (let i = 0; i <= 4; i++) {
    const val = (i / 4) * maxEntropy;
    ctx.fillText(val.toFixed(1), pad.left + pw + 8, pad.top + (1 - val / maxEntropy) * ph + 4);
  }
  ctx.save();
  ctx.translate(w - 8, pad.top + ph / 2);
  ctx.rotate(Math.PI / 2);
  ctx.textAlign = "center";
  ctx.fillText("avg entropy bits (dashed)", 0, 0);
  ctx.restore();

  // Legend
  let lx = pad.left;
  ctx.textAlign = "left";
  for (const [lang, conf] of Object.entries(LANG_CONFIG)) {
    if (!crossLingStats[lang]) continue;
    ctx.fillStyle = conf.color;
    ctx.fillRect(lx, pad.top - 20, 12, 3);
    ctx.font = "9px 'IBM Plex Mono', monospace";
    ctx.fillText(conf.label, lx + 16, pad.top - 16);
    lx += 80;
  }
}

// ── 5. Convergence Chart (REAL DATA) ─────────────────────────────────────

function initConvergenceChart(): void {
  const r = initCanvas("viz-convergence");
  if (!r) return;
  const { ctx, w, h } = r;

  const pad = { top: 30, right: 20, bottom: 40, left: 55 };
  const pw = w - pad.left - pad.right;
  const ph = h - pad.top - pad.bottom;

  ctx.fillStyle = COLORS.bg;
  ctx.fillRect(0, 0, w, h);

  if (!crossLingStats) {
    ctx.font = "12px 'IBM Plex Mono', monospace";
    ctx.fillStyle = COLORS.textDim; ctx.textAlign = "center";
    ctx.fillText("Cross-linguistic data not loaded", w / 2, h / 2);
    return;
  }

  const maxDepth = 12;

  // Compute cumulative log2(nodes) as a rough dimensionality proxy
  // This is H_cumulative = sum of shell entropies, which bounds the
  // number of independent bits needed to address all nodes up to depth d
  const dimCurves: Record<string, number[]> = {};
  let maxDim = 0;

  for (const [lang, stats] of Object.entries(crossLingStats)) {
    if (!LANG_CONFIG[lang]) continue;
    const dims: number[] = [0];
    for (let d = 1; d <= maxDepth; d++) {
      const s = stats.find(x => x.depth === d);
      if (!s) { dims.push(dims[dims.length - 1]); continue; }
      // log2(population) is a measure of addressing complexity
      const logPop = s.nodes > 0 ? Math.log2(s.nodes) : 0;
      dims.push(logPop);
    }
    dimCurves[lang] = dims;
    maxDim = Math.max(maxDim, ...dims);
  }
  maxDim = Math.ceil(maxDim / 5) * 5 + 2;

  // Grid
  ctx.strokeStyle = "rgba(255,255,255,0.04)";
  ctx.lineWidth = 0.5;
  for (let i = 0; i <= 5; i++) {
    const y = pad.top + (i / 5) * ph;
    ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(pad.left + pw, y); ctx.stroke();
  }

  // Convergence band: all languages peak in log2(pop) around 15-18
  const bandMin = 14;
  const bandMax = 18;
  const yMin = pad.top + (1 - bandMax / maxDim) * ph;
  const yMax = pad.top + (1 - bandMin / maxDim) * ph;
  ctx.fillStyle = "rgba(78,205,196,0.05)";
  ctx.fillRect(pad.left, yMin, pw, yMax - yMin);
  ctx.strokeStyle = "rgba(78,205,196,0.15)";
  ctx.lineWidth = 0.5;
  ctx.setLineDash([4, 4]);
  ctx.beginPath(); ctx.moveTo(pad.left, yMin); ctx.lineTo(pad.left + pw, yMin); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(pad.left, yMax); ctx.lineTo(pad.left + pw, yMax); ctx.stroke();
  ctx.setLineDash([]);

  ctx.font = "10px 'IBM Plex Mono', monospace";
  ctx.fillStyle = "rgba(78,205,196,0.3)";
  ctx.textAlign = "right";
  ctx.fillText("peak addressing complexity \u2248 2\u00B9\u2075\u207B\u00B9\u2078 distinct paths", pad.left + pw - 4, yMin + 14);

  // Lines
  for (const [lang, dims] of Object.entries(dimCurves)) {
    const conf = LANG_CONFIG[lang];
    if (!conf) continue;
    ctx.beginPath();
    dims.forEach((d, i) => {
      const x = pad.left + (i / maxDepth) * pw;
      const y = pad.top + (1 - d / maxDim) * ph;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.strokeStyle = hexAlpha(conf.color, 0.8);
    ctx.lineWidth = 2;
    ctx.stroke();

    // Points
    dims.forEach((d, i) => {
      const x = pad.left + (i / maxDepth) * pw;
      const y = pad.top + (1 - d / maxDim) * ph;
      ctx.beginPath();
      ctx.arc(x, y, 2.5, 0, Math.PI * 2);
      ctx.fillStyle = hexAlpha(conf.color, 0.6);
      ctx.fill();
    });
  }

  // X axis
  ctx.font = "10px 'IBM Plex Mono', monospace";
  ctx.fillStyle = COLORS.textDim;
  ctx.textAlign = "center";
  for (let d = 0; d <= maxDepth; d += 2) {
    ctx.fillText(d.toString(), pad.left + (d / maxDepth) * pw, pad.top + ph + 20);
  }
  ctx.fillText("phoneme depth", pad.left + pw / 2, pad.top + ph + 36);

  // Y axis
  ctx.textAlign = "right";
  for (let i = 0; i <= 5; i++) {
    const val = (i / 5) * maxDim;
    ctx.fillText(val.toFixed(0), pad.left - 8, pad.top + (1 - val / maxDim) * ph + 4);
  }
  ctx.save();
  ctx.translate(14, pad.top + ph / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = "center";
  ctx.fillText("log\u2082(shell population) \u2014 addressing bits", 0, 0);
  ctx.restore();

  // Legend
  const legend = document.getElementById("convergence-legend")!;
  legend.innerHTML = "";
  for (const [lang, conf] of Object.entries(LANG_CONFIG)) {
    if (!dimCurves[lang]) continue;
    const item = document.createElement("div");
    item.className = "legend-item";
    item.innerHTML = `<div class="legend-swatch" style="background: ${conf.color};"></div>${conf.label}`;
    legend.appendChild(item);
  }
}

// ── 6. Poincaré Disk ────────────────────────────────────────────────────

function initPoincareDisk(): void {
  const r = initCanvas("viz-poincare");
  if (!r) return;
  const { canvas, ctx, w, h } = r;

  const cx = w / 2;
  const cy = h / 2;
  const diskR = Math.min(w, h) / 2 - 40;

  // Use real trie structure if available: take depth <= 4 nodes
  interface PNode { hx: number; hy: number; depth: number; parentIdx: number; phoneme: string; role: string; count: number }
  const pnodes: PNode[] = [];

  if (essayData) {
    // Build from real trie: map 3D positions to 2D Poincaré disk
    const maxR3d = Math.max(...essayData.nodes.filter(n => n.d <= 4).map(n => Math.sqrt(n.x * n.x + n.z * n.z)));
    const nodeIdxMap = new Map<number, number>();

    // Root
    pnodes.push({ hx: 0, hy: 0, depth: 0, parentIdx: -1, phoneme: "ROOT", role: "m", count: essayData.meta.totalWords });
    nodeIdxMap.set(essayData.nodes[0].id, 0);

    // Add nodes depth 1-4
    for (const n of essayData.nodes) {
      if (n.d < 1 || n.d > 4) continue;
      // Project 3D to 2D and map into Poincaré disk
      const r3d = Math.sqrt(n.x * n.x + n.z * n.z);
      const angle = Math.atan2(n.z, n.x);
      // Map radius to Poincaré (tanh-like)
      const hr = Math.tanh((r3d / maxR3d) * 1.8);
      const idx = pnodes.length;
      nodeIdxMap.set(n.id, idx);
      pnodes.push({
        hx: Math.cos(angle) * hr,
        hy: Math.sin(angle) * hr,
        depth: n.d,
        parentIdx: n.pid !== null ? (nodeIdxMap.get(n.pid) ?? -1) : -1,
        phoneme: n.ph,
        role: n.role,
        count: n.cnt,
      });
    }
  } else {
    // Fallback: generate synthetic tree
    const rng = seedRandom(2718);
    pnodes.push({ hx: 0, hy: 0, depth: 0, parentIdx: -1, phoneme: "ROOT", role: "m", count: 1 });
    for (let i = 0; i < 5; i++) {
      const angle = (i / 5) * Math.PI * 2;
      const idx = pnodes.length;
      pnodes.push({ hx: Math.cos(angle) * 0.25, hy: Math.sin(angle) * 0.25, depth: 1, parentIdx: 0, phoneme: "", role: "o", count: 1 });
      for (let j = 0; j < 3; j++) {
        const a2 = angle + (j - 1) * 0.4;
        const jIdx = pnodes.length;
        pnodes.push({ hx: Math.cos(a2) * 0.55, hy: Math.sin(a2) * 0.55, depth: 2, parentIdx: idx, phoneme: "", role: "n", count: 1 });
        for (let k = 0; k < 2; k++) {
          const a3 = a2 + (k - 0.5) * 0.2 + (rng() - 0.5) * 0.1;
          pnodes.push({ hx: Math.cos(a3) * 0.8, hy: Math.sin(a3) * 0.8, depth: 3, parentIdx: jIdx, phoneme: "", role: "c", count: 1 });
        }
      }
    }
  }

  let rotation = 0;
  let dragging = false;
  let lastX = 0;

  function project(hx: number, hy: number): { x: number; y: number } {
    const cos = Math.cos(rotation), sin = Math.sin(rotation);
    const rx = hx * cos - hy * sin;
    const ry = hx * sin + hy * cos;
    return { x: cx + rx * diskR, y: cy + ry * diskR };
  }

  function draw(): void {
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = COLORS.bg;
    ctx.fillRect(0, 0, w, h);

    // Disk boundary
    ctx.beginPath();
    ctx.arc(cx, cy, diskR, 0, Math.PI * 2);
    ctx.strokeStyle = "rgba(255,255,255,0.08)";
    ctx.lineWidth = 1;
    ctx.stroke();

    // Geodesic circles
    for (let i = 1; i <= 4; i++) {
      const er = diskR * Math.tanh(i * 0.4);
      ctx.beginPath();
      ctx.arc(cx, cy, er, 0, Math.PI * 2);
      ctx.strokeStyle = "rgba(255,255,255,0.03)";
      ctx.lineWidth = 0.5;
      ctx.stroke();
    }

    // Edges
    for (const node of pnodes) {
      if (node.parentIdx < 0) continue;
      const parent = pnodes[node.parentIdx];
      const p1 = project(parent.hx, parent.hy);
      const p2 = project(node.hx, node.hy);
      const alpha = 0.03 + (1 - node.depth / 5) * 0.1;
      ctx.beginPath();
      ctx.moveTo(p1.x, p1.y);
      ctx.lineTo(p2.x, p2.y);
      const color = ROLE_COLORS[node.role] || COLORS.accent;
      ctx.strokeStyle = hexAlpha(color, alpha);
      ctx.lineWidth = 0.5;
      ctx.stroke();
    }

    // Nodes
    const maxCnt = Math.max(...pnodes.map(n => n.count));
    for (const node of pnodes) {
      const p = project(node.hx, node.hy);
      const brightness = node.depth === 0 ? 1 : Math.min(1, 0.2 + Math.log10(node.count + 1) / Math.log10(maxCnt + 1) * 0.8);
      const size = node.depth === 0 ? 5 : Math.max(0.8, 3.5 - node.depth * 0.6);
      const color = ROLE_COLORS[node.role] || COLORS.accent;
      ctx.beginPath();
      ctx.arc(p.x, p.y, size, 0, Math.PI * 2);
      ctx.fillStyle = hexAlpha(color, brightness * 0.7);
      ctx.fill();

      // Labels for depth 1
      if (node.depth === 1 && node.phoneme) {
        ctx.font = "9px 'IBM Plex Mono', monospace";
        ctx.fillStyle = hexAlpha(color, 0.5);
        ctx.textAlign = "center";
        ctx.fillText(node.phoneme, p.x, p.y - size - 4);
      }
    }

    ctx.font = "10px 'IBM Plex Mono', monospace";
    ctx.fillStyle = COLORS.textDim;
    ctx.textAlign = "center";
    ctx.fillText(`Poincar\u00E9 disk \u00B7 ${pnodes.length.toLocaleString()} nodes \u00B7 drag to rotate`, cx, h - 10);
  }

  canvas.addEventListener("mousedown", (e) => { dragging = true; lastX = e.clientX; canvas.style.cursor = "grabbing"; });
  canvas.addEventListener("mousemove", (e) => {
    if (!dragging) return;
    rotation += (e.clientX - lastX) * 0.005;
    lastX = e.clientX;
    draw();
  });
  canvas.addEventListener("mouseup", () => { dragging = false; canvas.style.cursor = "grab"; });
  canvas.addEventListener("mouseleave", () => { dragging = false; canvas.style.cursor = "grab"; });
  canvas.style.cursor = "grab";
  draw();
}

// ── 7. Voronoi Tessellation ──────────────────────────────────────────────

function initVoronoi(): void {
  const r = initCanvas("viz-voronoi");
  if (!r) return;
  const { canvas, ctx, w, h } = r;

  interface VPoint { x: number; y: number; color: string; label: string }

  const rng = seedRandom(1618);
  // Use real motifs as concept labels
  const conceptLabels = essayData
    ? essayData.meta.motifs.slice(0, 24).map(m => m.seq)
    : ["hot", "cold", "big", "small", "fast", "slow", "happy", "sad",
      "near", "far", "light", "dark", "up", "down", "old", "new",
      "hard", "soft", "wet", "dry", "loud", "quiet", "sweet", "bitter"];

  const points: VPoint[] = [];
  const colors = [COLORS.accent, COLORS.red, COLORS.yellow, COLORS.purple, COLORS.blue, COLORS.green];

  // Seed with initial points
  for (let i = 0; i < 12; i++) {
    points.push({
      x: 40 + rng() * (w - 80),
      y: 40 + rng() * (h - 80),
      color: colors[i % colors.length],
      label: conceptLabels[i] || "",
    });
  }

  function draw(): void {
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = COLORS.bg;
    ctx.fillRect(0, 0, w, h);

    if (points.length < 2) return;

    // Voronoi via nearest-point scan
    const cellSize = 3;
    for (let px = 0; px < w; px += cellSize) {
      for (let py = 0; py < h; py += cellSize) {
        let minDist = Infinity, nearest = 0, secondDist = Infinity;
        for (let i = 0; i < points.length; i++) {
          const dx = px - points[i].x, dy = py - points[i].y;
          const d = dx * dx + dy * dy;
          if (d < minDist) { secondDist = minDist; minDist = d; nearest = i; }
          else if (d < secondDist) { secondDist = d; }
        }
        ctx.fillStyle = hexAlpha(points[nearest].color, 0.06);
        ctx.fillRect(px, py, cellSize, cellSize);
        if (Math.sqrt(secondDist) - Math.sqrt(minDist) < 5) {
          ctx.fillStyle = hexAlpha(points[nearest].color, 0.18);
          ctx.fillRect(px, py, cellSize, cellSize);
        }
      }
    }

    // Points
    points.forEach((p) => {
      ctx.beginPath(); ctx.arc(p.x, p.y, 5, 0, Math.PI * 2);
      ctx.fillStyle = p.color; ctx.fill();
      ctx.beginPath(); ctx.arc(p.x, p.y, 10, 0, Math.PI * 2);
      ctx.strokeStyle = hexAlpha(p.color, 0.3); ctx.lineWidth = 1; ctx.stroke();
      if (p.label) {
        ctx.font = "10px 'IBM Plex Mono', monospace";
        ctx.fillStyle = hexAlpha(p.color, 0.7);
        ctx.textAlign = "center";
        ctx.fillText(p.label, p.x, p.y - 14);
      }
    });

    ctx.font = "10px 'IBM Plex Mono', monospace";
    ctx.fillStyle = COLORS.textDim;
    ctx.textAlign = "left";
    ctx.fillText(`${points.length} concept-points \u00B7 click to add`, 12, h - 10);
  }

  canvas.addEventListener("click", (e) => {
    const rect = canvas.getBoundingClientRect();
    points.push({
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
      color: colors[points.length % colors.length],
      label: conceptLabels[points.length % conceptLabels.length] || "",
    });
    draw();
  });
  canvas.style.cursor = "crosshair";
  draw();
}

// ── Populate data-driven HTML ────────────────────────────────────────────

function populateDataDrivenContent(): void {
  if (!essayData) return;

  // Fill in the stats summary in the hero or wherever needed
  const statsEl = document.getElementById("real-stats");
  if (statsEl) {
    statsEl.textContent = `${essayData.meta.totalNodes.toLocaleString()} nodes \u00B7 ${essayData.meta.totalTerminals.toLocaleString()} terminals \u00B7 ${essayData.meta.phonemeInventory.length} phonemes \u00B7 ${essayData.meta.totalWords.toLocaleString()} words`;
  }

  // Fill the convergence table with real data
  if (crossLingStats) {
    const tbody = document.getElementById("convergence-tbody");
    if (tbody) {
      tbody.innerHTML = "";
      for (const [lang, stats] of Object.entries(crossLingStats)) {
        const conf = LANG_CONFIG[lang];
        if (!conf) continue;
        // Find peak shell population
        const peakShell = stats.reduce((best, s) => s.nodes > best.nodes ? s : best, stats[0]);
        // Find depth where branching drops below 2
        const convergeDepth = stats.find(s => s.depth > 1 && s.avgBranch < 2)?.depth || "n/a";
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td style="color: ${conf.color};">${conf.label}</td>
          <td>${conf.family}</td>
          <td>${conf.type}</td>
          <td>${peakShell.nodes.toLocaleString()} (d=${peakShell.depth})</td>
          <td>${convergeDepth}</td>
        `;
        tbody.appendChild(tr);
      }
    }
  }
}

// ── Initialize ───────────────────────────────────────────────────────────

async function init(): Promise<void> {
  await loadData();

  initScrollEngine();
  initHeroBackground();
  initIPAViz();
  initTokenizer();
  initTrieBuilder();
  initRadialTrie();
  initEntropyChart();
  initConvergenceChart();
  initPoincareDisk();
  initVoronoi();
  populateDataDrivenContent();

  // Hide loading if present
  document.getElementById("loading")?.classList.add("hidden");
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
