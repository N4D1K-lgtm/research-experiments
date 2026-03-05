import type { TrieNodeData, TrieMetadata } from "../types";
import { getRoleColor, getRoleLabel } from "../color/languagePalette";

const el = () => document.getElementById("tooltip")!;

let cachedMetadata: TrieMetadata | null = null;

export function setTooltipMetadata(metadata: TrieMetadata): void {
  cachedMetadata = metadata;
}

export function showTooltip(node: TrieNodeData, x: number, y: number, fullPath?: string): void {
  const tooltip = el();
  const total = node.totalCount;

  // Phonological position badge
  const roleColor = getRoleColor(node.phonologicalPosition);
  const roleLabel = getRoleLabel(node.phonologicalPosition);
  const positionBadge = `<span style="display:inline-block;font-size:8px;font-weight:600;text-transform:uppercase;letter-spacing:0.8px;padding:1px 5px;border-radius:3px;background:${roleColor}22;color:${roleColor};vertical-align:middle;margin-left:4px">${roleLabel}</span>`;

  const terminalTag = node.isTerminal
    ? `<span class="terminal-tag">word</span>`
    : `<span class="passthrough-tag">path</span>`;

  // Transition probabilities
  let transitionHtml = "";
  if (node.transitionProbs) {
    const entries = Object.entries(node.transitionProbs).slice(0, 8);
    if (entries.length > 0) {
      const rows = entries.map(([phoneme, prob]) => {
        const pct = (prob * 100).toFixed(1);
        const barWidth = Math.min(100, prob * 100);
        return `
          <div class="lang-row">
            <span class="lang-name" style="font-family:monospace;min-width:30px">/${phoneme}/</span>
            <div class="lang-bar">
              <div class="lang-bar-fill" style="width:${barWidth}%;background:${roleColor}"></div>
            </div>
            <span class="lang-count">${pct}%</span>
          </div>
        `;
      }).join("");
      const totalChildren = Object.keys(node.transitionProbs).length;
      const moreLabel = totalChildren > 8 ? ` <span style="color:#555;font-size:9px">+${totalChildren - 8} more</span>` : "";
      transitionHtml = `
        <div style="margin-top:8px;padding-top:6px;border-top:1px solid rgba(255,255,255,0.06)">
          <div style="font-size:9px;color:#555;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px">Transitions (${totalChildren} paths)${moreLabel}</div>
          ${rows}
        </div>
      `;
    }
  }

  // Motifs
  let motifHtml = "";
  if (node.motifs && node.motifs.length > 0) {
    const motifTags = node.motifs.slice(0, 5).map((m) =>
      `<span style="display:inline-block;font-size:9px;padding:1px 4px;border-radius:3px;background:rgba(255,204,0,0.15);color:#cc9900;margin:1px 2px;font-family:monospace">/${m}/</span>`
    ).join("");
    const more = node.motifs.length > 5 ? ` <span style="color:#555;font-size:9px">+${node.motifs.length - 5}</span>` : "";
    motifHtml = `<div style="margin-top:6px"><span style="font-size:9px;color:#555;text-transform:uppercase;letter-spacing:0.5px">Motifs: </span>${motifTags}${more}</div>`;
  }

  // Allophones with distributional context
  let allophoneHtml = "";
  if (node.allophones && node.allophones.length > 0) {
    const forms = node.allophones.map((a) => {
      let contextStr = "";
      if (cachedMetadata?.allophoneContexts?.[a]) {
        const ctx = cachedMetadata.allophoneContexts[a];
        const before = ctx.before.slice(0, 3).map((p) => `/${p}/`).join(" ");
        const after = ctx.after.slice(0, 3).map((p) => `/${p}/`).join(" ");
        contextStr = ` <span style="color:#555;font-size:9px">${before} _ ${after}</span>`;
      }
      return `<span class="allophone">[${a}]</span>${contextStr}`;
    }).join(" ");
    allophoneHtml = `<div class="allophone-row">allophones: ${forms}</div>`;
  }

  // Words (for terminal nodes)
  let wordsHtml = "";
  if (node.isTerminal && node.words) {
    const wordEntries: string[] = [];
    for (const [lang, ws] of Object.entries(node.words)) {
      if (!ws || ws.length === 0) continue;
      const termCount = node.terminalCounts?.[lang] ?? ws.length;
      const wordList = ws.map((w) => `<em>${w}</em>`).join(", ");
      const more = termCount > ws.length ? ` +${termCount - ws.length} more` : "";
      wordEntries.push(
        `<div class="word-row">${wordList}<span class="word-more">${more}</span></div>`
      );
    }
    if (wordEntries.length > 0) {
      wordsHtml = `<div class="words-section">${wordEntries.join("")}</div>`;
    }
  }

  tooltip.innerHTML = `
    <div class="phoneme" style="color:${roleColor}">/${fullPath ?? node.phoneme}/ ${terminalTag} ${positionBadge}</div>
    <div class="meta">depth ${node.depth} · ${total.toLocaleString()} paths · ${node.childCount} branches</div>
    ${allophoneHtml}
    ${motifHtml}
    ${transitionHtml}
    ${wordsHtml}
  `;
  tooltip.style.display = "block";

  const pad = 16;
  const rect = tooltip.getBoundingClientRect();
  let tx = x + pad;
  let ty = y + pad;
  if (tx + rect.width > window.innerWidth) tx = x - rect.width - pad;
  if (ty + rect.height > window.innerHeight) ty = y - rect.height - pad;
  tooltip.style.left = `${tx}px`;
  tooltip.style.top = `${ty}px`;
}

export function hideTooltip(): void {
  el().style.display = "none";
}
