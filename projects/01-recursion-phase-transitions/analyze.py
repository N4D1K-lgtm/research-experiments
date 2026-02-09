#!/usr/bin/env python3
"""
Strange Loops - A1 analysis and visualization.

Usage:
    python3 analyze.py                    # all plots + stats
    python3 analyze.py --query "SQL"      # ad-hoc query
    python3 analyze.py --plot cascade     # specific plot
    python3 analyze.py --stats-only       # just numbers
    python3 analyze.py --latex-tables     # emit LaTeX table fragments
"""
import argparse
import math
import sqlite3
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.lines import Line2D

DB_PATH = Path(__file__).parent / "strange_loops.db"
OUT_DIR = Path(__file__).parent / "plots"


# ── helpers ──────────────────────────────────────────────────────

def connect():
    if not DB_PATH.exists():
        print(f"Database not found: {DB_PATH}", file=sys.stderr)
        sys.exit(1)
    return sqlite3.connect(str(DB_PATH))


def query(conn, sql, params=()):
    cur = conn.execute(sql, params)
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()
    return cols, rows


def print_table(cols, rows, max_rows=50):
    widths = [len(c) for c in cols]
    for row in rows[:max_rows]:
        for i, v in enumerate(row):
            widths[i] = max(widths[i], len(str(v)))
    header = " | ".join(f"{c:>{w}}" for c, w in zip(cols, widths))
    sep = "-+-".join("-" * w for w in widths)
    print(header)
    print(sep)
    for row in rows[:max_rows]:
        print(" | ".join(f"{str(v):>{w}}" for v, w in zip(row, widths)))
    if len(rows) > max_rows:
        print(f"  ... ({len(rows) - max_rows} more rows)")


def experiment_info(conn, exp_id):
    _, rows = query(conn, "SELECT name, basis FROM experiments WHERE id=?", (exp_id,))
    if rows:
        return rows[0]
    return (f"exp#{exp_id}", "[]")


# ── label and classification ─────────────────────────────────────

LABEL_MAP = {
    "baseline_ski":       "Baseline (SKI)",
    # Singles
    "extended_ski_ss":    "SKI + SS",
    "extended_ski_sk":    "SKI + SK",
    "extended_ski_si":    "SKI + SI",
    "extended_ski_sss":   "SKI + S(SS)",
    "extended_ski_ks":    "SKI + KS",
    "extended_ski_kk":    "SKI + KK",
    "extended_ski_ki":    "SKI + KI",
    "extended_ski_sii":   "SKI + SII",
    "extended_ski_skk":   "SKI + SKK",
    "extended_ski_sks":   "SKI + S(KS)",
    "extended_ski_ski2":  "SKI + S(KI)",
    # Pairs
    "combo_ss_sss":       "SS + S(SS)",
    "combo_ss_ks":        "SS + KS",
    "combo_ss_sii":       "SS + SII",
    "combo_sss_sii":      "S(SS) + SII",
    # Triples
    "combo_ss_sss_sii":   "SS + S(SS) + SII",
    "combo_ss_sk_ks":     "SS + SK + KS",
}

# Classification metadata for each single motif experiment
MOTIF_META = {
    "extended_ski_ss":   {"head": "S", "motif_size": 2, "motifs": ["SS"]},
    "extended_ski_sk":   {"head": "S", "motif_size": 2, "motifs": ["SK"]},
    "extended_ski_si":   {"head": "S", "motif_size": 2, "motifs": ["SI"]},
    "extended_ski_sss":  {"head": "S", "motif_size": 3, "motifs": ["S(SS)"]},
    "extended_ski_ks":   {"head": "K", "motif_size": 2, "motifs": ["KS"]},
    "extended_ski_kk":   {"head": "K", "motif_size": 2, "motifs": ["KK"]},
    "extended_ski_ki":   {"head": "K", "motif_size": 2, "motifs": ["KI"]},
    "extended_ski_sii":  {"head": "S", "motif_size": 3, "motifs": ["SII"]},
    "extended_ski_skk":  {"head": "S", "motif_size": 3, "motifs": ["SKK"]},
    "extended_ski_sks":  {"head": "S", "motif_size": 3, "motifs": ["S(KS)"]},
    "extended_ski_ski2": {"head": "S", "motif_size": 3, "motifs": ["S(KI)"]},
}

# Which single-motif experiments belong to each combo
COMBO_COMPONENTS = {
    "combo_ss_sss":      ["extended_ski_ss", "extended_ski_sss"],
    "combo_ss_ks":       ["extended_ski_ss", "extended_ski_ks"],
    "combo_ss_sii":      ["extended_ski_ss", "extended_ski_sii"],
    "combo_sss_sii":     ["extended_ski_sss", "extended_ski_sii"],
    "combo_ss_sss_sii":  ["extended_ski_ss", "extended_ski_sss", "extended_ski_sii"],
    "combo_ss_sk_ks":    ["extended_ski_ss", "extended_ski_sk", "extended_ski_ks"],
}


def short_label(name):
    """Compact experiment label for plots."""
    return LABEL_MAP.get(name, name)


def load_reports(conn, exp_id):
    _, rows = query(conn,
        "SELECT size, total_terms, distinct_nfs, compression_ratio, "
        "divergent_count, explosive_count, motif_count, avg_steps, max_steps "
        "FROM reports WHERE experiment_id=? ORDER BY size", (exp_id,))
    return rows


def all_experiment_ids(conn):
    _, rows = query(conn, "SELECT DISTINCT id FROM experiments ORDER BY id")
    return [r[0] for r in rows]


def exp_name(conn, eid):
    name, _ = experiment_info(conn, eid)
    return name


def linreg(xs, ys):
    """Simple linear regression. Returns (slope, intercept, r^2)."""
    n = len(xs)
    sx, sy = sum(xs), sum(ys)
    sxx = sum(x*x for x in xs)
    sxy = sum(x*y for x, y in zip(xs, ys))
    syy = sum(y*y for y in ys)
    denom = n * sxx - sx * sx
    if denom == 0:
        return 0, 0, 0
    b = (n * sxy - sx * sy) / denom
    a = (sy - b * sx) / n
    ss_res = sum((y - (a + b*x))**2 for x, y in zip(xs, ys))
    ss_tot = sum((y - sy/n)**2 for y in ys)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return b, a, r2


# ── color scheme ─────────────────────────────────────────────────

PALETTE = {
    "baseline_ski":       "#2c3e50",  # dark blue-grey
    # Original singles
    "extended_ski_ss":    "#e74c3c",  # red
    "extended_ski_sk":    "#3498db",  # blue
    "extended_ski_si":    "#2ecc71",  # green
    "extended_ski_sss":   "#9b59b6",  # purple
    "extended_ski_ks":    "#e67e22",  # orange
    # New singles
    "extended_ski_kk":    "#f39c12",  # gold
    "extended_ski_ki":    "#d35400",  # dark orange
    "extended_ski_sii":   "#c0392b",  # dark red
    "extended_ski_skk":   "#1abc9c",  # teal
    "extended_ski_sks":   "#8e44ad",  # dark purple
    "extended_ski_ski2":  "#16a085",  # dark teal
    # Pairs
    "combo_ss_sss":       "#e74c3c",  # red (strong)
    "combo_ss_ks":        "#95a5a6",  # grey
    "combo_ss_sii":       "#c0392b",  # dark red
    "combo_sss_sii":      "#8e44ad",  # dark purple
    # Triples
    "combo_ss_sss_sii":   "#2c3e50",  # dark
    "combo_ss_sk_ks":     "#7f8c8d",  # grey
}

def color_for(conn, eid):
    name = exp_name(conn, eid)
    return PALETTE.get(name, "#999999")


def is_single(name):
    return name.startswith("extended_ski_")

def is_combo(name):
    return name.startswith("combo_")


# ── figure 1: compression ratio decay ────────────────────────────

def plot_compression_decay(conn):
    """Compression ratio vs size for all experiments, with exponential fits."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    exp_ids = all_experiment_ids(conn)

    # Only plot baseline + singles (combos are separate in fig11)
    fits = {}
    for eid in exp_ids:
        rows = load_reports(conn, eid)
        name = exp_name(conn, eid)
        if is_combo(name):
            continue
        label = short_label(name)
        color = color_for(conn, eid)

        sizes = [r[0] for r in rows]
        ratios = [r[3] for r in rows]
        lw = 2.5 if "baseline" in name else 1.8
        marker = "o" if "baseline" in name else "s"
        ax.plot(sizes, ratios, f"{marker}-", color=color, linewidth=lw,
                markersize=7, label=label, zorder=3 if "baseline" in name else 2)

        # Exponential fit on log(ratio) for sizes where ratio < 1
        fit_s = [s for s, r in zip(sizes, ratios) if r < 0.99]
        fit_r = [r for r in ratios if r < 0.99]
        if len(fit_s) >= 2:
            b, a, r2 = linreg(fit_s, [math.log(r) for r in fit_r])
            fits[name] = (b, math.exp(a), r2)
            xs = np.linspace(min(fit_s), max(fit_s) + 0.5, 50)
            ys = np.exp(a + b * xs)
            ax.plot(xs, ys, "--", color=color, alpha=0.4, linewidth=1)

    ax.set_xlabel("Term size $N$ (number of combinator leaves)", fontsize=12)
    ax.set_ylabel("Compression ratio $\\rho(N) = D(N)/T(N)$", fontsize=12)
    ax.set_title("Compression Ratio Decay", fontsize=14, fontweight="bold")
    ax.set_yscale("log")
    ax.set_xticks(range(1, 9))
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=8, loc="upper right", ncol=2)

    # Annotate fit for baseline
    if "baseline_ski" in fits:
        b, _, r2 = fits["baseline_ski"]
        ax.text(0.02, 0.02,
                f"Baseline: $\\rho \\sim e^{{{b:.3f} \\cdot N}}$ ($R^2={r2:.3f}$)",
                transform=ax.transAxes, fontsize=9, color=PALETTE["baseline_ski"],
                verticalalignment="bottom")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig1_compression_decay.png", dpi=200)
    fig.savefig(OUT_DIR / "fig1_compression_decay.pdf")
    plt.close(fig)
    print(f"  -> fig1_compression_decay")
    return fits


# ── figure 2: abstraction gap ────────────────────────────────────

def plot_abstraction_gap(conn):
    """Total terms vs distinct NFs on log scale."""
    fig, ax = plt.subplots(figsize=(9, 5.5))

    # Show baseline + best and worst extended
    show = ["baseline_ski", "extended_ski_ss", "extended_ski_sss", "extended_ski_ks"]
    exp_ids = all_experiment_ids(conn)

    for eid in exp_ids:
        name = exp_name(conn, eid)
        if name not in show:
            continue
        rows = load_reports(conn, eid)
        sizes = [r[0] for r in rows]
        totals = [r[1] for r in rows]
        nfs = [r[2] for r in rows]
        color = color_for(conn, eid)
        label = short_label(name)

        ax.plot(sizes, totals, "s--", color=color, alpha=0.35, markersize=5)
        ax.plot(sizes, nfs, "o-", color=color, linewidth=2, markersize=7,
                label=f"{label}")

    # Reference line: total terms for baseline
    rows_b = load_reports(conn, 1)
    sizes_b = [r[0] for r in rows_b]
    totals_b = [r[1] for r in rows_b]
    ax.plot(sizes_b, totals_b, "k--", alpha=0.3, linewidth=1, label="Total terms (baseline)")
    ax.fill_between(sizes_b,
                     [r[2] for r in rows_b], totals_b,
                     alpha=0.06, color=PALETTE["baseline_ski"])

    ax.set_xlabel("Term size $N$", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("The Abstraction Gap", fontsize=14, fontweight="bold")
    ax.set_yscale("log")
    ax.set_xticks(range(1, 9))
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig2_abstraction_gap.png", dpi=200)
    fig.savefig(OUT_DIR / "fig2_abstraction_gap.pdf")
    plt.close(fig)
    print(f"  -> fig2_abstraction_gap")


# ── figure 3: divergence/explosion onset ─────────────────────────

def plot_divergence(conn):
    """Divergent + explosive percentage by size for all experiments."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    exp_ids = all_experiment_ids(conn)

    for eid in exp_ids:
        name = exp_name(conn, eid)
        if is_combo(name):
            continue
        rows = load_reports(conn, eid)
        color = color_for(conn, eid)
        label = short_label(name)

        sizes = [r[0] for r in rows]
        totals = [r[1] for r in rows]
        divg = [r[4] for r in rows]
        expl = [r[5] for r in rows]
        non_normal_pct = [100*(d + e)/t if t > 0 else 0
                          for d, e, t in zip(divg, expl, totals)]
        divg_pct = [100*d/t if t > 0 else 0 for d, t in zip(divg, totals)]

        lw = 2 if "baseline" in name else 1.5
        ax1.plot(sizes, divg_pct, "o-", color=color, linewidth=lw,
                 markersize=5, label=label)
        ax2.plot(sizes, non_normal_pct, "o-", color=color, linewidth=lw,
                 markersize=5, label=label)

    ax1.set_xlabel("Term size $N$", fontsize=11)
    ax1.set_ylabel("Divergent (%)", fontsize=11)
    ax1.set_title("Divergence Rate", fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.2)
    ax1.legend(fontsize=7, ncol=2)

    ax2.set_xlabel("Term size $N$", fontsize=11)
    ax2.set_ylabel("Non-normalizing (%)", fontsize=11)
    ax2.set_title("Total Non-normalizing Rate", fontsize=13, fontweight="bold")
    ax2.grid(True, alpha=0.2)
    ax2.legend(fontsize=7, ncol=2)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig3_divergence.png", dpi=200)
    fig.savefig(OUT_DIR / "fig3_divergence.pdf")
    plt.close(fig)
    print(f"  -> fig3_divergence")


# ── figure 4: NF multiplicity distribution (Zipf check) ──────────

def plot_nf_distribution(conn, exp_id=1, size=7):
    """Log-log plot of NF multiplicities."""
    _, rows = query(conn,
        "SELECT normal_form, COUNT(*) as c FROM reductions "
        "WHERE experiment_id=? AND size=? AND status='normal' "
        "GROUP BY normal_form ORDER BY c DESC", (exp_id, size))
    if not rows:
        return

    counts = sorted([r[1] for r in rows], reverse=True)
    ranks = list(range(1, len(counts) + 1))
    top_nfs = [(r[0], r[1]) for r in rows[:15]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Left: rank-frequency (Zipf)
    ax1.scatter(ranks, counts, s=8, alpha=0.5, color=PALETTE["baseline_ski"])
    # Fit power law on top portion
    top_n = min(len(ranks), 500)
    b, a, r2 = linreg([math.log(r) for r in ranks[:top_n]],
                       [math.log(c) for c in counts[:top_n]])
    xs = np.array(ranks[:top_n])
    ax1.plot(xs, np.exp(a) * xs**b, "r-", alpha=0.7, linewidth=1.5,
             label=f"Power law: $\\alpha={-b:.2f}$ ($R^2={r2:.3f}$)")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Rank", fontsize=11)
    ax1.set_ylabel("Multiplicity (input terms $\\to$ this NF)", fontsize=11)
    ax1.set_title(f"NF Rank-Frequency (size {size})", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.2)

    # Right: top 15 attractors
    labels = [nf[:25] for nf, _ in top_nfs]
    values = [c for _, c in top_nfs]
    y_pos = range(len(labels))
    ax2.barh(y_pos, values, color=PALETTE["baseline_ski"], alpha=0.8)
    ax2.set_yticks(list(y_pos))
    ax2.set_yticklabels(labels, fontsize=9, family="monospace")
    ax2.invert_yaxis()
    ax2.set_xlabel("Input terms mapping to this NF", fontsize=11)
    ax2.set_title(f"Top Attractor Normal Forms (size {size})",
                  fontsize=13, fontweight="bold")
    ax2.grid(True, alpha=0.2, axis="x")

    fig.tight_layout()
    fig.savefig(OUT_DIR / f"fig4_nf_distribution_s{size}.png", dpi=200)
    fig.savefig(OUT_DIR / f"fig4_nf_distribution_s{size}.pdf")
    plt.close(fig)
    print(f"  -> fig4_nf_distribution_s{size}")
    return -b, r2  # Zipf exponent


# ── figure 5: motif reuse value ──────────────────────────────────

def plot_motif_reuse(conn, exp_id=1):
    """Top motifs by reuse savings across sizes."""
    _, rows = query(conn,
        "SELECT size, term, nf_count, savings FROM motifs "
        "WHERE experiment_id=? ORDER BY size, savings DESC", (exp_id,))
    if not rows:
        return

    by_size = {}
    for size, term, nf_count, savings in rows:
        by_size.setdefault(size, []).append((term, nf_count, savings))

    fig, ax = plt.subplots(figsize=(10, 5.5))
    sizes = sorted(by_size.keys())

    # Track top motifs for consistent coloring
    all_motifs = []
    for s in sizes:
        for term, _, _ in by_size[s][:5]:
            if term not in all_motifs:
                all_motifs.append(term)
    cmap = plt.get_cmap("tab10")
    motif_colors = {m: cmap(i) for i, m in enumerate(all_motifs[:10])}

    bar_w = 0.14
    for s in sizes:
        entries = by_size[s][:5]
        for j, (term, nf_count, savings) in enumerate(entries):
            color = motif_colors.get(term, "#999")
            x = s + (j - 2) * bar_w
            ax.bar(x, savings, width=bar_w, color=color, alpha=0.85,
                   label=term if s == sizes[0] else "")

    ax.set_xlabel("Term size $N$", fontsize=12)
    ax.set_ylabel("Reuse savings (size units)", fontsize=12)
    ax.set_title("Top Motif Reuse Value by Size (Baseline)", fontsize=14, fontweight="bold")
    ax.set_xticks(sizes)
    ax.grid(True, alpha=0.2, axis="y")

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=10, loc="upper left")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig5_motif_reuse.png", dpi=200)
    fig.savefig(OUT_DIR / "fig5_motif_reuse.pdf")
    plt.close(fig)
    print(f"  -> fig5_motif_reuse")


# ── figure 6: cross-motif comparison ─────────────────────────────

def plot_motif_comparison(conn):
    """Compression ratio advantage of each named motif vs baseline."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    base_eid = None
    ext_eids = []
    for eid in all_experiment_ids(conn):
        name = exp_name(conn, eid)
        if "baseline" in name:
            base_eid = eid
        elif is_single(name):
            ext_eids.append(eid)

    if base_eid is None or not ext_eids:
        return

    base_rows = load_reports(conn, base_eid)
    base_ratios = {r[0]: r[3] for r in base_rows}

    # Left: compression ratio for each experiment
    for eid in [base_eid] + ext_eids:
        rows = load_reports(conn, eid)
        name = exp_name(conn, eid)
        color = color_for(conn, eid)
        label = short_label(name)
        sizes = [r[0] for r in rows]
        ratios = [r[3] for r in rows]
        lw = 2.5 if "baseline" in name else 1.5
        ax1.plot(sizes, ratios, "o-", color=color, linewidth=lw,
                 markersize=6, label=label)

    ax1.set_yscale("log")
    ax1.set_xlabel("Term size $N$", fontsize=11)
    ax1.set_ylabel("Compression ratio $\\rho(N)$", fontsize=11)
    ax1.set_title("Compression Ratio by Motif", fontsize=13, fontweight="bold")
    ax1.set_xticks(range(1, 8))
    ax1.grid(True, alpha=0.2)
    ax1.legend(fontsize=7, ncol=2)

    # Right: ratio advantage (ext/base) at each size
    for eid in ext_eids:
        rows = load_reports(conn, eid)
        name = exp_name(conn, eid)
        color = color_for(conn, eid)
        label = short_label(name)
        sizes = [r[0] for r in rows]
        advantages = []
        for r in rows:
            br = base_ratios.get(r[0], 0)
            advantages.append(r[3] / br if br > 0 else 1.0)
        ax2.plot(sizes, advantages, "o-", color=color, linewidth=1.5,
                 markersize=6, label=label)

    ax2.axhline(y=1.0, color="black", linestyle="--", alpha=0.3, linewidth=1)
    ax2.set_xlabel("Term size $N$", fontsize=11)
    ax2.set_ylabel("$\\rho_{ext}/\\rho_{base}$ (advantage)", fontsize=11)
    ax2.set_title("Expressiveness Advantage over Baseline",
                  fontsize=13, fontweight="bold")
    ax2.set_xticks(range(1, 8))
    ax2.grid(True, alpha=0.2)
    ax2.legend(fontsize=7, ncol=2)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig6_motif_comparison.png", dpi=200)
    fig.savefig(OUT_DIR / "fig6_motif_comparison.pdf")
    plt.close(fig)
    print(f"  -> fig6_motif_comparison")


# ── figure 7: step distribution ──────────────────────────────────

def plot_step_distribution(conn, exp_id=1, size=7):
    """Histogram of reduction steps for normalizing terms."""
    _, rows = query(conn,
        "SELECT steps, COUNT(*) FROM reductions "
        "WHERE experiment_id=? AND size=? AND status='normal' "
        "GROUP BY steps ORDER BY steps", (exp_id, size))
    if not rows:
        return

    steps = [r[0] for r in rows]
    counts = [r[1] for r in rows]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(steps, counts, color=PALETTE["baseline_ski"], alpha=0.8, edgecolor="white",
           linewidth=0.3)
    ax.set_xlabel("Reduction steps to normal form", fontsize=12)
    ax.set_ylabel("Number of terms", fontsize=12)
    ax.set_title(f"Reduction Step Distribution (Baseline, size {size})",
                 fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.2, axis="y")

    # Annotate mean
    _, rpt = query(conn, "SELECT avg_steps, max_steps FROM reports WHERE experiment_id=? AND size=?",
                   (exp_id, size))
    if rpt:
        avg, mx = rpt[0]
        ax.axvline(avg, color="#e74c3c", linestyle="--", linewidth=1.5, alpha=0.7)
        ax.text(avg + 0.3, max(counts)*0.9, f"mean={avg:.1f}", color="#e74c3c", fontsize=10)

    fig.tight_layout()
    fig.savefig(OUT_DIR / f"fig7_steps_s{size}.png", dpi=200)
    fig.savefig(OUT_DIR / f"fig7_steps_s{size}.pdf")
    plt.close(fig)
    print(f"  -> fig7_steps_s{size}")


# ── figure 8: motif ranking bar chart ────────────────────────────

def plot_motif_ranking(conn):
    """Bar chart ranking all single motifs by compression advantage at size 6."""
    base_eid = None
    ext_data = []
    for eid in all_experiment_ids(conn):
        name = exp_name(conn, eid)
        if "baseline" in name:
            base_eid = eid
        elif is_single(name):
            ext_data.append((eid, name))

    if base_eid is None:
        return

    base_rows = load_reports(conn, base_eid)
    base_r6 = next((r[3] for r in base_rows if r[0] == 6), None)
    if base_r6 is None:
        return

    rankings = []
    for eid, name in ext_data:
        rows = load_reports(conn, eid)
        r6 = next((r for r in rows if r[0] == 6), None)
        if r6:
            advantage = r6[3] / base_r6 if base_r6 > 0 else 0
            head = MOTIF_META.get(name, {}).get("head", "?")
            rankings.append((short_label(name), r6[3], advantage, r6[2], name, head))

    rankings.sort(key=lambda x: -x[2])

    fig, ax = plt.subplots(figsize=(10, 5.5))
    labels = [r[0] for r in rankings]
    advantages = [r[2] for r in rankings]
    heads = [r[5] for r in rankings]
    colors = ["#e74c3c" if h == "S" else "#3498db" for h in heads]

    bars = ax.barh(range(len(labels)), advantages, color=colors, alpha=0.85)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.axvline(x=1.0, color="black", linestyle="--", alpha=0.3)
    ax.set_xlabel("Compression advantage $\\rho_{ext}/\\rho_{base}$ at $N=6$", fontsize=12)
    ax.set_title("Motif Ranking by Expressiveness Gain", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.2, axis="x")

    for i, (lbl, cr, adv, nfs, _, head) in enumerate(rankings):
        ax.text(adv + 0.02, i, f"{adv:.2f}x ({nfs:,} NFs)", va="center", fontsize=9)

    # Legend for head type
    legend_elements = [
        Line2D([0], [0], color="#e74c3c", lw=8, alpha=0.85, label="S-headed"),
        Line2D([0], [0], color="#3498db", lw=8, alpha=0.85, label="K-headed"),
    ]
    ax.legend(handles=legend_elements, fontsize=10, loc="lower right")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig8_motif_ranking.png", dpi=200)
    fig.savefig(OUT_DIR / "fig8_motif_ranking.pdf")
    plt.close(fig)
    print(f"  -> fig8_motif_ranking")


# ── figure 9: the cascade (annotated timeline) ──────────────────

def plot_cascade(conn):
    """Multi-metric cascade showing what emerges at each size."""
    rows = load_reports(conn, 1)  # baseline
    if not rows:
        return

    sizes = [r[0] for r in rows]
    ratios = [r[3] for r in rows]
    divg = [r[4] for r in rows]
    expl = [r[5] for r in rows]
    motifs = [r[6] for r in rows]
    avg_steps = [r[7] for r in rows]

    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

    # Panel 1: Compression ratio
    ax = axes[0]
    ax.plot(sizes, ratios, "o-", color=PALETTE["baseline_ski"], linewidth=2, markersize=8)
    ax.set_ylabel("$\\rho(N)$", fontsize=12)
    ax.set_title("The Abstraction Cascade (Baseline SKI)", fontsize=14, fontweight="bold")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.2)
    ax.annotate("Compression\nonset", xy=(3, ratios[2]), xytext=(3.5, 0.8),
                arrowprops=dict(arrowstyle="->", color="#666"), fontsize=9, color="#666")

    # Panel 2: Divergence + explosion
    ax = axes[1]
    ax.bar([s - 0.15 for s in sizes], divg, width=0.3,
           color="#e74c3c", alpha=0.8, label="Divergent")
    ax.bar([s + 0.15 for s in sizes], expl, width=0.3,
           color="#e67e22", alpha=0.8, label="Explosive")
    ax.set_ylabel("Count", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    if any(d > 0 for d in divg):
        first_div = next(s for s, d in zip(sizes, divg) if d > 0)
        ax.annotate(f"First divergence\n(size {first_div})",
                    xy=(first_div, divg[sizes.index(first_div)]),
                    xytext=(first_div + 0.5, max(d for d in divg if d > 0) * 0.8),
                    arrowprops=dict(arrowstyle="->", color="#c00"),
                    fontsize=9, color="#c00")

    # Panel 3: New motifs
    ax = axes[2]
    ax.plot(sizes, motifs, "o-", color="#9b59b6", linewidth=2, markersize=8)
    ax.set_ylabel("New motifs", fontsize=12)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.2)

    # Panel 4: Average steps
    ax = axes[3]
    ax.plot(sizes, avg_steps, "o-", color="#2ecc71", linewidth=2, markersize=8)
    ax.set_ylabel("Avg steps", fontsize=12)
    ax.set_xlabel("Term size $N$", fontsize=12)
    ax.grid(True, alpha=0.2)
    ax.set_xticks(range(1, max(sizes) + 1))

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig9_cascade.png", dpi=200)
    fig.savefig(OUT_DIR / "fig9_cascade.pdf")
    plt.close(fig)
    print(f"  -> fig9_cascade")


# ── figure 10: classification scatter ────────────────────────────

def plot_classification_scatter(conn):
    """Scatter plot: advantage vs non-normalizing rate, colored by head type."""
    base_eid = None
    for eid in all_experiment_ids(conn):
        name = exp_name(conn, eid)
        if "baseline" in name:
            base_eid = eid
            break

    if base_eid is None:
        return

    base_rows = load_reports(conn, base_eid)
    base_r6 = next((r for r in base_rows if r[0] == 6), None)
    if base_r6 is None:
        return
    base_ratio = base_r6[3]

    fig, ax = plt.subplots(figsize=(9, 6))

    for eid in all_experiment_ids(conn):
        name = exp_name(conn, eid)
        if not is_single(name):
            continue
        meta = MOTIF_META.get(name)
        if meta is None:
            continue

        rows = load_reports(conn, eid)
        r6 = next((r for r in rows if r[0] == 6), None)
        if r6 is None:
            continue

        advantage = r6[3] / base_ratio if base_ratio > 0 else 0
        non_norm_pct = 100 * (r6[4] + r6[5]) / r6[1] if r6[1] > 0 else 0

        color = "#e74c3c" if meta["head"] == "S" else "#3498db"
        marker = "o" if meta["motif_size"] == 2 else "^"
        ax.scatter(non_norm_pct, advantage, c=color, marker=marker, s=120,
                   zorder=5, edgecolors="white", linewidth=0.8)
        # Label
        label = short_label(name).replace("SKI + ", "")
        ax.annotate(label, (non_norm_pct, advantage),
                    textcoords="offset points", xytext=(8, 4),
                    fontsize=9, color=color)

    ax.axhline(y=1.0, color="black", linestyle="--", alpha=0.3, linewidth=1)
    ax.set_xlabel("Non-normalizing rate at $N=6$ (%)", fontsize=12)
    ax.set_ylabel("Compression advantage ($\\rho_{ext}/\\rho_{base}$)", fontsize=12)
    ax.set_title("Motif Classification: Expressiveness vs Divergence",
                 fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.2)

    # Legend
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#e74c3c",
               markersize=10, label="S-headed, size 2"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#e74c3c",
               markersize=10, label="S-headed, size 3"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#3498db",
               markersize=10, label="K-headed, size 2"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="upper left")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig10_classification.png", dpi=200)
    fig.savefig(OUT_DIR / "fig10_classification.pdf")
    plt.close(fig)
    print(f"  -> fig10_classification")


# ── figure 11: combination interaction effects ───────────────────

def plot_combo_interactions(conn):
    """Expected vs actual advantage for combination experiments."""
    base_eid = None
    for eid in all_experiment_ids(conn):
        name = exp_name(conn, eid)
        if "baseline" in name:
            base_eid = eid
            break

    if base_eid is None:
        return

    base_rows = load_reports(conn, base_eid)
    base_r6 = next((r for r in base_rows if r[0] == 6), None)
    base_r5 = next((r for r in base_rows if r[0] == 5), None)
    if base_r6 is None or base_r5 is None:
        return

    # Collect single-motif advantages at both sizes
    single_advantages = {}  # name -> {size: advantage}
    for eid in all_experiment_ids(conn):
        name = exp_name(conn, eid)
        if not is_single(name):
            continue
        rows = load_reports(conn, eid)
        single_advantages[name] = {}
        for sz, base_r in [(6, base_r6), (5, base_r5)]:
            r = next((r for r in rows if r[0] == sz), None)
            if r and base_r[3] > 0:
                single_advantages[name][sz] = r[3] / base_r[3]

    # Collect combo results
    combos = []
    for eid in all_experiment_ids(conn):
        name = exp_name(conn, eid)
        if not is_combo(name):
            continue
        components = COMBO_COMPONENTS.get(name, [])
        if not components:
            continue

        rows = load_reports(conn, eid)
        # Pick comparison size based on whether it's a triple (size 5) or pair (size 6)
        if len(components) == 3:
            compare_size = 5
            base_r = base_r5
        else:
            compare_size = 6
            base_r = base_r6

        r = next((r for r in rows if r[0] == compare_size), None)
        if r is None or base_r[3] <= 0:
            continue

        actual = r[3] / base_r[3]

        # Expected: product of individual advantages (independence assumption)
        expected = 1.0
        for comp in components:
            sa = single_advantages.get(comp, {}).get(compare_size, 1.0)
            expected *= sa

        combos.append((short_label(name), actual, expected, compare_size))

    if not combos:
        print("  -> fig11_combo_interactions (no combo data yet)")
        return

    fig, ax = plt.subplots(figsize=(9, 6))

    # Diagonal (expected = actual)
    all_vals = [c[1] for c in combos] + [c[2] for c in combos]
    lo, hi = min(all_vals) * 0.8, max(all_vals) * 1.2
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.3, linewidth=1, label="Expected = Actual")

    for label, actual, expected, sz in combos:
        color = "#e74c3c" if actual > expected else "#3498db"
        marker = "o" if sz == 6 else "^"
        ax.scatter(expected, actual, c=color, marker=marker, s=150,
                   zorder=5, edgecolors="white", linewidth=1)
        ax.annotate(label, (expected, actual),
                    textcoords="offset points", xytext=(8, 6),
                    fontsize=9)

    ax.set_xlabel("Expected advantage (product of singles)", fontsize=12)
    ax.set_ylabel("Actual advantage", fontsize=12)
    ax.set_title("Combination Interaction Effects", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.2)

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#e74c3c",
               markersize=10, label="Synergistic (actual > expected)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#3498db",
               markersize=10, label="Redundant (actual < expected)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#999",
               markersize=8, label="Pairs (size 6)"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#999",
               markersize=8, label="Triples (size 5)"),
    ]
    ax.legend(handles=legend_elements, fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig11_combo_interactions.png", dpi=200)
    fig.savefig(OUT_DIR / "fig11_combo_interactions.pdf")
    plt.close(fig)
    print(f"  -> fig11_combo_interactions")


# ── figure 12: motif taxonomy heatmap ────────────────────────────

def plot_motif_taxonomy(conn):
    """Heatmap of all single motifs x metrics."""
    base_eid = None
    for eid in all_experiment_ids(conn):
        name = exp_name(conn, eid)
        if "baseline" in name:
            base_eid = eid
            break

    if base_eid is None:
        return

    base_rows = load_reports(conn, base_eid)
    base_r6 = next((r for r in base_rows if r[0] == 6), None)
    if base_r6 is None:
        return

    # Collect data for all single motifs
    motif_data = []
    for eid in all_experiment_ids(conn):
        name = exp_name(conn, eid)
        if not is_single(name):
            continue
        meta = MOTIF_META.get(name)
        if meta is None:
            continue

        rows = load_reports(conn, eid)
        r6 = next((r for r in rows if r[0] == 6), None)
        if r6 is None:
            continue

        advantage = r6[3] / base_r6[3] if base_r6[3] > 0 else 0
        divg_pct = 100 * r6[4] / r6[1] if r6[1] > 0 else 0
        expl_pct = 100 * r6[5] / r6[1] if r6[1] > 0 else 0
        non_norm_pct = divg_pct + expl_pct

        label = short_label(name).replace("SKI + ", "")
        motif_data.append({
            "label": label,
            "head": meta["head"],
            "size": meta["motif_size"],
            "advantage": advantage,
            "distinct_nfs": r6[2],
            "divg_pct": divg_pct,
            "expl_pct": expl_pct,
            "non_norm_pct": non_norm_pct,
            "avg_steps": r6[7],
        })

    if not motif_data:
        print("  -> fig12_taxonomy (no data)")
        return

    # Sort: S-headed first (by advantage desc), then K-headed (by advantage desc)
    motif_data.sort(key=lambda d: (0 if d["head"] == "S" else 1, -d["advantage"]))

    labels = [d["label"] for d in motif_data]
    metrics = ["advantage", "distinct_nfs", "non_norm_pct", "avg_steps"]
    metric_labels = ["Advantage", "Distinct NFs", "Non-norm %", "Avg Steps"]

    # Build matrix (normalize each metric to [0, 1] for the heatmap)
    raw = np.array([[d[m] for m in metrics] for d in motif_data])
    # Normalize per column
    col_min = raw.min(axis=0)
    col_max = raw.max(axis=0)
    col_range = col_max - col_min
    col_range[col_range == 0] = 1
    normed = (raw - col_min) / col_range

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(normed, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(metric_labels)))
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)

    # Add text annotations with raw values
    for i in range(len(labels)):
        for j in range(len(metrics)):
            val = raw[i, j]
            if metrics[j] == "distinct_nfs":
                txt = f"{int(val):,}"
            elif metrics[j] in ("divg_pct", "expl_pct", "non_norm_pct"):
                txt = f"{val:.1f}%"
            elif metrics[j] == "advantage":
                txt = f"{val:.2f}x"
            else:
                txt = f"{val:.2f}"
            text_color = "white" if normed[i, j] < 0.3 or normed[i, j] > 0.8 else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9, color=text_color)

    # Color bars on the left for head type
    for i, d in enumerate(motif_data):
        color = "#e74c3c" if d["head"] == "S" else "#3498db"
        ax.add_patch(plt.Rectangle((-0.6, i - 0.4), 0.15, 0.8, color=color, clip_on=False))

    ax.set_title("Motif Taxonomy", fontsize=14, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.6, label="Normalized value")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig12_taxonomy.png", dpi=200)
    fig.savefig(OUT_DIR / "fig12_taxonomy.pdf")
    plt.close(fig)
    print(f"  -> fig12_taxonomy")


# ── quantitative analysis ────────────────────────────────────────

def stats(conn):
    """Print quantitative analysis."""
    print("\n" + "=" * 70)
    print("QUANTITATIVE ANALYSIS")
    print("=" * 70)

    exp_ids = all_experiment_ids(conn)

    for eid in exp_ids:
        name = exp_name(conn, eid)
        label = short_label(name)
        rows = load_reports(conn, eid)
        if not rows:
            continue

        print(f"\n-- {label} --")

        sizes = [r[0] for r in rows]
        ratios = [r[3] for r in rows]
        nfs = [r[2] for r in rows]
        totals = [r[1] for r in rows]
        divg = [r[4] for r in rows]

        # Compression ratio decay rate
        fit_s = [s for s, r in zip(sizes, ratios) if r < 0.99]
        fit_r = [r for r in ratios if r < 0.99]
        if len(fit_s) >= 2:
            b, a, r2 = linreg(fit_s, [math.log(r) for r in fit_r])
            half_life = -math.log(2) / b if b < 0 else float("inf")
            print(f"  Compression decay: rho ~ e^({b:.3f} * N), R^2={r2:.3f}")
            print(f"  Half-life: {half_life:.2f} size steps")
            print(f"  Per-step decay: {math.exp(b):.4f}")

        # NF growth rate
        if len(nfs) >= 2:
            gr = [nfs[i] / nfs[i - 1] for i in range(1, len(nfs)) if nfs[i - 1] > 0]
            if gr:
                print(f"  NF growth factor: {np.mean(gr):.2f}x (range {min(gr):.2f}-{max(gr):.2f})")

        # First divergence
        for s, d in zip(sizes, divg):
            if d > 0:
                pct = 100 * d / totals[sizes.index(s)]
                print(f"  First divergence: size {s} ({d} terms, {pct:.2f}%)")
                break

        # First explosion
        for r in rows:
            if r[5] > 0:
                pct = 100 * r[5] / r[1]
                print(f"  First explosion: size {r[0]} ({r[5]} terms, {pct:.2f}%)")
                break

    # Cross-motif comparison at size 6
    base_rows = load_reports(conn, 1)
    base_r6 = next((r[3] for r in base_rows if r[0] == 6), None)
    if base_r6:
        print(f"\n-- Single motif ranking at size 6 (baseline rho={base_r6:.4f}) --")
        print(f"  {'Motif':>14} | {'Head':>4} | {'rho':>8} | {'Advantage':>10} | {'Distinct NFs':>12} | {'Divg%':>6} | {'Expl%':>6}")
        print(f"  {'-'*80}")
        ranking = []
        for eid in exp_ids:
            name = exp_name(conn, eid)
            if not is_single(name):
                continue
            rows = load_reports(conn, eid)
            r6 = next((r for r in rows if r[0] == 6), None)
            if r6:
                adv = r6[3] / base_r6
                dp = 100 * r6[4] / r6[1] if r6[1] > 0 else 0
                ep = 100 * r6[5] / r6[1] if r6[1] > 0 else 0
                head = MOTIF_META.get(name, {}).get("head", "?")
                ranking.append((short_label(name), head, r6[3], adv, r6[2], dp, ep))
        ranking.sort(key=lambda x: -x[3])
        for lbl, head, rho, adv, nfs, dp, ep in ranking:
            print(f"  {lbl:>14} | {head:>4} | {rho:>8.4f} | {adv:>9.4f}x | {nfs:>12,} | {dp:>5.1f}% | {ep:>5.1f}%")

    # Combo comparison
    combo_data = []
    for eid in exp_ids:
        name = exp_name(conn, eid)
        if not is_combo(name):
            continue
        rows = load_reports(conn, eid)
        # Pick latest available size
        if rows:
            r = rows[-1]
            sz = r[0]
            base_r = next((br for br in base_rows if br[0] == sz), None)
            if base_r and base_r[3] > 0:
                adv = r[3] / base_r[3]
                dp = 100 * r[4] / r[1] if r[1] > 0 else 0
                ep = 100 * r[5] / r[1] if r[1] > 0 else 0
                combo_data.append((short_label(name), sz, r[3], adv, r[2], dp, ep))

    if combo_data:
        print(f"\n-- Combination experiments --")
        print(f"  {'Combo':>22} | {'N':>2} | {'rho':>8} | {'Advantage':>10} | {'Distinct NFs':>12} | {'Divg%':>6} | {'Expl%':>6}")
        print(f"  {'-'*85}")
        combo_data.sort(key=lambda x: -x[3])
        for lbl, sz, rho, adv, nfs, dp, ep in combo_data:
            print(f"  {lbl:>22} | {sz:>2} | {rho:>8.4f} | {adv:>9.4f}x | {nfs:>12,} | {dp:>5.1f}% | {ep:>5.1f}%")

    # Top attractors
    print(f"\n-- Top attractor NFs (baseline, size 7) --")
    cols, arows = query(conn,
        "SELECT normal_form, COUNT(*) as c FROM reductions "
        "WHERE experiment_id=1 AND size=7 AND status='normal' "
        "GROUP BY normal_form ORDER BY c DESC LIMIT 15", ())
    print_table(cols, arows)


# ── LaTeX table fragments ────────────────────────────────────────

def latex_tables(conn):
    """Emit LaTeX-ready table fragments."""

    # Table 1: Baseline survey
    print("\n% Table: Baseline SKI survey")
    print("\\begin{tabular}{r r r r r r r r}")
    print("\\toprule")
    print("$N$ & $T(N)$ & $D(N)$ & $\\rho(N)$ & Divg & Expl & Motifs & $\\overline{s}$ \\\\")
    print("\\midrule")
    rows = load_reports(conn, 1)
    for r in rows:
        size, total, nfs, ratio, divg, expl, motifs, avg_s, max_s = r
        print(f"{size} & {total:,} & {nfs:,} & {ratio:.4f} & {divg} & {expl} & {motifs:,} & {avg_s:.2f} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}\n")

    # Table 2: All single motifs at size 6
    exp_ids = all_experiment_ids(conn)
    base_rows = load_reports(conn, 1)
    base_r6 = next((r for r in base_rows if r[0] == 6), None)

    print("% Table: All single motifs at N=6")
    print("\\begin{tabular}{l l r r r r r}")
    print("\\toprule")
    print("Motif & Head & $D(6)$ & $\\rho(6)$ & Advantage & Divg\\% & Expl\\% \\\\")
    print("\\midrule")

    if base_r6:
        print(f"Baseline (SKI) & -- & {base_r6[2]:,} & {base_r6[3]:.4f} & $1.000\\times$ & "
              f"{100*base_r6[4]/base_r6[1]:.1f}\\% & {100*base_r6[5]/base_r6[1]:.1f}\\% \\\\")

    ranking = []
    for eid in exp_ids:
        name = exp_name(conn, eid)
        if not is_single(name):
            continue
        rows = load_reports(conn, eid)
        r6 = next((r for r in rows if r[0] == 6), None)
        if r6 and base_r6:
            adv = r6[3] / base_r6[3]
            head = MOTIF_META.get(name, {}).get("head", "?")
            ranking.append((short_label(name), head, r6, adv))
    ranking.sort(key=lambda x: -x[3])

    for lbl, head, r, adv in ranking:
        dp = 100 * r[4] / r[1] if r[1] > 0 else 0
        ep = 100 * r[5] / r[1] if r[1] > 0 else 0
        print(f"{lbl} & {head} & {r[2]:,} & {r[3]:.4f} & {adv:.3f}$\\times$ & {dp:.1f}\\% & {ep:.1f}\\% \\\\")
    print("\\bottomrule")
    print("\\end{tabular}\n")

    # Table 3: Combination experiments
    print("% Table: Combination experiments")
    print("\\begin{tabular}{l r r r r r r}")
    print("\\toprule")
    print("Combination & $N$ & $k$ & $D(N)$ & $\\rho(N)$ & Advantage & Non-norm\\% \\\\")
    print("\\midrule")

    for eid in exp_ids:
        name = exp_name(conn, eid)
        if not is_combo(name):
            continue
        rows = load_reports(conn, eid)
        if not rows:
            continue
        r = rows[-1]
        sz = r[0]
        components = COMBO_COMPONENTS.get(name, [])
        k = 3 + len(components)
        base_r = next((br for br in base_rows if br[0] == sz), None)
        if base_r and base_r[3] > 0:
            adv = r[3] / base_r[3]
            nn = 100 * (r[4] + r[5]) / r[1] if r[1] > 0 else 0
            lbl = short_label(name)
            print(f"{lbl} & {sz} & {k} & {r[2]:,} & {r[3]:.4f} & {adv:.3f}$\\times$ & {nn:.1f}\\% \\\\")
    print("\\bottomrule")
    print("\\end{tabular}\n")

    # Table 4: Top attractors
    print("% Table: Top attractor normal forms (baseline, size 7)")
    print("\\begin{tabular}{l r r}")
    print("\\toprule")
    print("Normal Form & Multiplicity & \\% of normalizing terms \\\\")
    print("\\midrule")
    _, arows = query(conn,
        "SELECT normal_form, COUNT(*) as c FROM reductions "
        "WHERE experiment_id=1 AND size=7 AND status='normal' "
        "GROUP BY normal_form ORDER BY c DESC LIMIT 15", ())
    _, total_row = query(conn,
        "SELECT COUNT(*) FROM reductions WHERE experiment_id=1 AND size=7 AND status='normal'", ())
    total_n = total_row[0][0] if total_row else 1
    for nf, c in arows:
        pct = 100 * c / total_n
        print(f"\\texttt{{{nf}}} & {c:,} & {pct:.1f}\\% \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")


# ── main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyze strange-loops experiment data")
    parser.add_argument("--query", "-q", help="Run an ad-hoc SQL query")
    parser.add_argument("--plot", "-p", help="Specific plot name")
    parser.add_argument("--size", type=int, default=7, help="Size for per-size plots")
    parser.add_argument("--exp", type=int, default=1, help="Experiment ID")
    parser.add_argument("--stats-only", action="store_true", help="Only print stats")
    parser.add_argument("--latex-tables", action="store_true", help="Emit LaTeX tables")
    args = parser.parse_args()

    conn = connect()

    if args.query:
        cols, rows = query(conn, args.query)
        print_table(cols, rows)
        return

    if args.latex_tables:
        latex_tables(conn)
        return

    OUT_DIR.mkdir(exist_ok=True)

    if not args.stats_only:
        plot_name = args.plot

        plots = {
            "compression": lambda: plot_compression_decay(conn),
            "gap": lambda: plot_abstraction_gap(conn),
            "divergence": lambda: plot_divergence(conn),
            "nf_dist": lambda: plot_nf_distribution(conn, args.exp, args.size),
            "motifs": lambda: plot_motif_reuse(conn, args.exp),
            "comparison": lambda: plot_motif_comparison(conn),
            "steps": lambda: plot_step_distribution(conn, args.exp, args.size),
            "ranking": lambda: plot_motif_ranking(conn),
            "cascade": lambda: plot_cascade(conn),
            "classification": lambda: plot_classification_scatter(conn),
            "combos": lambda: plot_combo_interactions(conn),
            "taxonomy": lambda: plot_motif_taxonomy(conn),
        }

        if plot_name:
            if plot_name in plots:
                print(f"Generating {plot_name}...")
                plots[plot_name]()
            else:
                print(f"Unknown plot: {plot_name}. Available: {', '.join(plots.keys())}")
        else:
            for name, fn in plots.items():
                print(f"Generating {name}...")
                fn()

    stats(conn)
    conn.close()


if __name__ == "__main__":
    main()
