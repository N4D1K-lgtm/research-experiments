import type { FilterState, TrieMetadata, PhonologicalPosition } from "../types";
import { ROLES, getRoleColor } from "../color/languagePalette";

export type FilterChangeCallback = (state: FilterState) => void;

export class FilterPanel {
  private state: FilterState;
  private onChange: FilterChangeCallback;
  private debounceTimer: number | null = null;

  constructor(metadata: TrieMetadata, onChange: FilterChangeCallback) {
    this.onChange = onChange;
    this.state = {
      maxDepth: metadata.maxDepth,
      minFrequency: 50,
      terminalsOnly: false,
      highlightMotifs: new Set(),
      positionFilter: new Set(),
    };
    this.render(metadata);
  }

  getState(): FilterState {
    return this.state;
  }

  private emit(): void {
    if (this.debounceTimer != null) clearTimeout(this.debounceTimer);
    this.debounceTimer = window.setTimeout(() => {
      this.onChange({
        ...this.state,
        highlightMotifs: new Set(this.state.highlightMotifs),
        positionFilter: new Set(this.state.positionFilter),
      });
    }, 100);
  }

  private render(metadata: TrieMetadata): void {
    const panel = document.getElementById("filter-panel")!;

    // Phonological position filter
    const posSection = document.createElement("div");
    posSection.className = "filter-section";
    posSection.innerHTML = "<h3>Phonological Role</h3>";

    for (const role of ROLES) {
      const label = document.createElement("label");
      const cb = document.createElement("input");
      cb.type = "checkbox";
      cb.checked = false; // unchecked = show all
      cb.addEventListener("change", () => {
        if (cb.checked) {
          this.state.positionFilter.add(role.position);
        } else {
          this.state.positionFilter.delete(role.position);
        }
        this.emit();
      });

      const dot = document.createElement("span");
      dot.style.cssText = `display:inline-block;width:8px;height:8px;border-radius:50%;background:${getRoleColor(role.position)};flex-shrink:0;box-shadow:0 0 6px ${getRoleColor(role.position)}44`;

      label.appendChild(cb);
      label.appendChild(dot);
      label.appendChild(document.createTextNode(` ${role.label}`));
      posSection.appendChild(label);
    }
    panel.appendChild(posSection);

    // Terminals only toggle
    const termSection = document.createElement("div");
    termSection.className = "filter-section";
    termSection.innerHTML = "<h3>Display</h3>";
    const termLabel = document.createElement("label");
    const termCb = document.createElement("input");
    termCb.type = "checkbox";
    termCb.checked = false;
    termCb.addEventListener("change", () => {
      this.state.terminalsOnly = termCb.checked;
      this.emit();
    });
    termLabel.appendChild(termCb);
    termLabel.appendChild(document.createTextNode(" Word endpoints only"));
    termSection.appendChild(termLabel);
    panel.appendChild(termSection);

    // Depth slider
    const depthSection = document.createElement("div");
    depthSection.className = "filter-section";
    depthSection.innerHTML = "<h3>Max Depth</h3>";
    const depthRow = document.createElement("div");
    depthRow.className = "slider-row";
    const depthSlider = document.createElement("input");
    depthSlider.type = "range";
    depthSlider.min = "1";
    depthSlider.max = String(metadata.maxDepth);
    depthSlider.value = String(metadata.maxDepth);
    const depthVal = document.createElement("span");
    depthVal.className = "value";
    depthVal.textContent = String(metadata.maxDepth);
    depthSlider.addEventListener("input", () => {
      this.state.maxDepth = parseInt(depthSlider.value);
      depthVal.textContent = depthSlider.value;
      this.emit();
    });
    depthRow.appendChild(document.createTextNode("1"));
    depthRow.appendChild(depthSlider);
    depthRow.appendChild(depthVal);
    depthSection.appendChild(depthRow);
    panel.appendChild(depthSection);

    // Frequency slider
    const freqSection = document.createElement("div");
    freqSection.className = "filter-section";
    freqSection.innerHTML = "<h3>Min Frequency</h3>";
    const freqRow = document.createElement("div");
    freqRow.className = "slider-row";
    const freqSlider = document.createElement("input");
    freqSlider.type = "range";
    freqSlider.min = "1";
    freqSlider.max = "5000";
    freqSlider.value = "50";
    const freqVal = document.createElement("span");
    freqVal.className = "value";
    freqVal.textContent = "50";
    freqSlider.addEventListener("input", () => {
      this.state.minFrequency = parseInt(freqSlider.value);
      freqVal.textContent = freqSlider.value;
      this.emit();
    });
    freqRow.appendChild(document.createTextNode("1"));
    freqRow.appendChild(freqSlider);
    freqRow.appendChild(freqVal);
    freqSection.appendChild(freqRow);
    panel.appendChild(freqSection);

    // Motif highlight
    if (metadata.motifs && metadata.motifs.length > 0) {
      const motifSection = document.createElement("div");
      motifSection.className = "filter-section";
      motifSection.innerHTML = "<h3>Highlight Motif</h3>";

      // Show top motifs as checkboxes
      const topMotifs = metadata.motifs.slice(0, 12);
      for (const motif of topMotifs) {
        const label = document.createElement("label");
        label.style.fontSize = "11px";
        const cb = document.createElement("input");
        cb.type = "checkbox";
        cb.checked = false;
        cb.addEventListener("change", () => {
          if (cb.checked) {
            this.state.highlightMotifs.add(motif.label);
          } else {
            this.state.highlightMotifs.delete(motif.label);
          }
          this.emit();
        });

        const countSpan = document.createElement("span");
        countSpan.style.cssText = "font-size:9px;color:#555;margin-left:auto;font-variant-numeric:tabular-nums";
        countSpan.textContent = motif.count.toLocaleString();

        label.appendChild(cb);
        label.appendChild(document.createTextNode(` /${motif.label}/`));
        label.appendChild(countSpan);
        motifSection.appendChild(label);
      }
      panel.appendChild(motifSection);
    }

    // Legend
    const legend = document.createElement("div");
    legend.className = "legend";
    legend.innerHTML = `
      <div class="legend-row">
        <div class="legend-swatch" style="background:#39a5c9"></div>
        <span>Onset (consonant)</span>
      </div>
      <div class="legend-row">
        <div class="legend-swatch" style="background:#c9a539"></div>
        <span>Nucleus (vowel)</span>
      </div>
      <div class="legend-row">
        <div class="legend-swatch" style="background:#9539c9"></div>
        <span>Coda (consonant)</span>
      </div>
      <div class="legend-row">
        <div class="legend-swatch" style="background:#39c980"></div>
        <span>Mixed role</span>
      </div>
      <div class="legend-row" style="margin-top:4px">
        <div class="legend-swatch" style="background:rgba(255,255,255,0.6)"></div>
        <span>Glow = word endpoint</span>
      </div>
      <div class="legend-row">
        <div class="legend-swatch" style="background:rgba(255,255,255,0.25);height:6px"></div>
        <span>Large = high entropy</span>
      </div>
      <div class="legend-row">
        <div class="legend-swatch" style="background:linear-gradient(90deg,rgba(255,255,255,0.04),rgba(255,255,255,0.5));height:2px"></div>
        <span>Bright edge = common</span>
      </div>
    `;
    panel.appendChild(legend);
  }
}
