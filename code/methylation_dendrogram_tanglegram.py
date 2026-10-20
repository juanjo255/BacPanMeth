#!/usr/bin/env python3
"""
Genome-level methylation dendrogram & tanglegram analysis.

For each replicon type (Chromosome / pSymA / pSymB / accessory plasmids)
  • build a genome × motif binary presence-absence matrix
  • cluster genomes by Jaccard distance (average linkage)
  • draw a labelled dendrogram

Then for every pair of replicon types draw a *tanglegram*:
  two facing dendrograms connected by lines whose colour encodes
  whether the genome ordering is concordant (blue) or discordant (red).
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from matplotlib.path import Path
import matplotlib.patheffects as pe
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, cophenet, leaves_list
from scipy.stats import spearmanr
from itertools import combinations

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE    = "/n/projects/jp2992/Plasmidome_project_MOLNG_4509"
MATRIX  = os.path.join(BASE, "methyl_call", "methylation_correlation",
                        "contig_motif_matrix.csv")
TAGS    = os.path.join(BASE, "contigs_to_tags.tsv")
OUT_DIR = os.path.join(BASE, "methyl_call", "methylation_correlation",
                        "dendrograms")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load & annotate motif matrix ───────────────────────────────────────────────
print("Loading motif matrix …")
df_raw = pd.read_csv(MATRIX, index_col=0)
FEAT = [c for c in df_raw.columns
        if c not in ("sample", "length", "category")
        and not c.endswith("_density")
        and not c.startswith("tsne")]

tags = pd.read_csv(TAGS, sep="\t", header=None, names=["contig", "replicon"])
tag_map = dict(zip(tags["contig"], tags["replicon"]))
df_raw["replicon"] = df_raw.index.map(tag_map)

# ── Build per-genome matrix for each replicon ─────────────────────────────────
print("Building per-genome matrices …")

REPLICONS = ["Chromosome", "pSymA", "pSymB", "Other"]
LABELS    = {"Chromosome": "Chromosome", "pSymA": "pSymA (megaplasmid)",
             "pSymB": "pSymB (megaplasmid)", "Other": "Accessory plasmids"}
COLORS    = {"Chromosome": "#e74c3c", "pSymA": "#3498db",
             "pSymB": "#f39c12", "Other": "#2ecc71"}

genome_matrices = {}   # replicon → DataFrame(index=sample, columns=motifs)

for rep in REPLICONS:
    sub = df_raw[df_raw["replicon"] == rep].copy()

    if rep == "Other":
        # Multiple accessory plasmids per sample → union (OR) presence
        agg = sub.groupby("sample")[FEAT].max()   # max of 0/1 = union
    else:
        # One contig per sample
        sub = sub.copy()
        sub.index = sub["sample"]
        agg = sub[FEAT]

    # Remove all-zero motif columns (no genome has that motif for this replicon)
    agg = agg.loc[:, agg.sum() > 0]

    # Keep only motifs present in ≥2 genomes (reduce noise)
    agg = agg.loc[:, agg.sum() >= 2]

    genome_matrices[rep] = agg.copy()
    print(f"  {rep:12s}: {len(agg)} genomes × {len(agg.columns)} motifs")

# ── Consistent genome colour palette ──────────────────────────────────────────
all_samples = sorted({s for df in genome_matrices.values() for s in df.index})
cmap = plt.cm.tab20
genome_colors = {s: cmap(i / max(len(all_samples)-1, 1))
                 for i, s in enumerate(all_samples)}

# ── Helper: Jaccard linkage + dendrogram data ─────────────────────────────────
def compute_linkage(mat: pd.DataFrame):
    """Return (Z, labels) using Jaccard distance, average linkage."""
    if len(mat) < 2:
        return None, list(mat.index)
    if mat.values.sum() == 0:          # all-zero matrix
        return None, list(mat.index)
    dist = pdist(mat.values.astype(float), metric="jaccard")
    dist = np.nan_to_num(dist, nan=1.0)
    Z = linkage(dist, method="average")
    return Z, list(mat.index)


def draw_dendrogram_ax(ax, Z, labels, title, replicon_color,
                       orientation="right", label_colors=None,
                       show_labels=True):
    """
    Draw a single dendrogram on `ax`.
    orientation='right' → leaves on right (for left panel of tanglegram)
    orientation='left'  → leaves on left  (for right panel)
    """
    if Z is None:
        ax.set_title(title, color=replicon_color, fontweight="bold")
        ax.axis("off")
        return {}

    ddata = dendrogram(
        Z,
        labels=labels,
        orientation=orientation,
        ax=ax,
        color_threshold=0,
        above_threshold_color="#555555",
        no_labels=not show_labels,
    )
    # Colour leaf labels by genome
    ax.set_title(title, color=replicon_color, fontweight="bold", fontsize=9)
    if show_labels and label_colors:
        axis = ax.get_yaxis() if orientation in ("right", "left") else ax.get_xaxis()
        for lbl in (ax.get_yticklabels() if orientation in ("right", "left")
                    else ax.get_xticklabels()):
            lbl.set_color(label_colors.get(lbl.get_text(), "black"))
            lbl.set_fontsize(7)
    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return ddata


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 – Four dendrograms (one per replicon) on a single figure
# ══════════════════════════════════════════════════════════════════════════════
print("\nDrawing replicon dendrograms …")

fig, axes = plt.subplots(1, 4, figsize=(22, 11))
linkages = {}

for ax, rep in zip(axes, REPLICONS):
    mat  = genome_matrices[rep]
    Z, labels = compute_linkage(mat)
    linkages[rep] = (Z, labels)

    if Z is not None:
        cc, _ = cophenet(Z, pdist(mat.values.astype(float), metric="jaccard"))
    else:
        cc = float("nan")

    draw_dendrogram_ax(
        ax, Z, labels,
        title=f"{LABELS[rep]}\n(n={len(mat)}, CCC={cc:.2f})",
        replicon_color=COLORS[rep],
        orientation="top",
        label_colors=genome_colors,
        show_labels=True,
    )

plt.suptitle("Genome clustering by methylation motif presence-absence\n"
             "(Jaccard distance, average linkage)", fontsize=12, y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig1_replicon_dendrograms.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT_DIR, "fig1_replicon_dendrograms.png"),
            dpi=180, bbox_inches="tight")
plt.close()
print("  fig1 saved.")


# ══════════════════════════════════════════════════════════════════════════════
# TANGLEGRAM FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def get_leaf_order(Z, labels):
    """Return the leaf labels in dendrogram display order (bottom→top)."""
    if Z is None:
        return labels
    order_idx = leaves_list(Z)
    return [labels[i] for i in order_idx]


def bezier_connect(ax, y_left, y_right, x_left, x_right,
                   color, alpha=0.6, lw=1.2):
    """Draw a cubic Bezier between two y-positions on opposing sides."""
    xmid = (x_left + x_right) / 2
    verts = [(x_left,  y_left),
             (xmid,    y_left),
             (xmid,    y_right),
             (x_right, y_right)]
    codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
    path  = Path(verts, codes)
    patch = mpatches.PathPatch(path, facecolor="none", edgecolor=color,
                               alpha=alpha, lw=lw)
    ax.add_patch(patch)


def tanglegram(Z_left, labels_left, Z_right, labels_right,
               title_left, title_right,
               color_left, color_right,
               genome_colors, shared_only=True):
    """
    Draw a tanglegram: two horizontal dendrograms facing each other,
    connected by Bezier curves.

    Lines are coloured:
      blue  → genome keeps the same *rank* between the two trees (concordant)
      red   → rank changes (discordant)
    Line opacity ∝ |rank change|.
    """
    # ── find shared genomes ──────────────────────────────────────────────────
    set_l, set_r = set(labels_left), set(labels_right)
    shared = sorted(set_l & set_r)
    if len(shared) < 2:
        return None

    # Re-filter matrices to shared genomes only
    idx_l = [labels_left.index(s)  for s in shared]
    idx_r = [labels_right.index(s) for s in shared]
    labels_left_sh  = [labels_left[i]  for i in idx_l]
    labels_right_sh = [labels_right[i] for i in idx_r]

    # Re-compute linkage on shared subset
    mat_l = genome_matrices[title_left.split()[0]]   # keyed by replicon name
    mat_r = genome_matrices[title_right.split()[0]]
    mat_l = mat_l.loc[shared]
    mat_r = mat_r.loc[shared]

    Z_l, _ = compute_linkage(mat_l)
    Z_r, _ = compute_linkage(mat_r)

    order_l = get_leaf_order(Z_l, shared) if Z_l is not None else shared
    order_r = get_leaf_order(Z_r, shared) if Z_r is not None else shared

    n = len(shared)

    # ── set up figure ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, max(8, n * 0.38)))
    ax_l   = fig.add_axes([0.02, 0.05, 0.38, 0.88])   # left dendrogram
    ax_mid = fig.add_axes([0.40, 0.05, 0.18, 0.88])   # connector space
    ax_r   = fig.add_axes([0.58, 0.05, 0.38, 0.88])   # right dendrogram

    # ── draw left dendrogram (leaves on right) ────────────────────────────────
    if Z_l is not None:
        ddata_l = dendrogram(Z_l, labels=shared, orientation="right",
                             ax=ax_l, color_threshold=0,
                             above_threshold_color="#555555", no_labels=False)
    else:
        ax_l.set_yticks(range(n))
        ax_l.set_yticklabels(order_l)
        ddata_l = {}
    ax_l.set_title(title_left, color=color_left, fontweight="bold", fontsize=10)
    for lbl in ax_l.get_yticklabels():
        lbl.set_color(genome_colors.get(lbl.get_text(), "black"))
        lbl.set_fontsize(7)
    ax_l.tick_params(length=0)
    for sp in ax_l.spines.values(): sp.set_visible(False)
    ax_l.set_xlabel("")

    # ── draw right dendrogram (leaves on left) ────────────────────────────────
    if Z_r is not None:
        ddata_r = dendrogram(Z_r, labels=shared, orientation="left",
                             ax=ax_r, color_threshold=0,
                             above_threshold_color="#555555", no_labels=False)
    else:
        ax_r.set_yticks(range(n))
        ax_r.set_yticklabels(order_r)
        ddata_r = {}
    ax_r.set_title(title_right, color=color_right, fontweight="bold", fontsize=10)
    for lbl in ax_r.get_yticklabels():
        lbl.set_color(genome_colors.get(lbl.get_text(), "black"))
        lbl.set_fontsize(7)
    ax_r.tick_params(length=0)
    for sp in ax_r.spines.values(): sp.set_visible(False)
    ax_r.set_xlabel("")

    # ── get leaf y-positions from the dendrogram ivl order ───────────────────
    # scipy dendrogram: ivl = leaf labels in order from bottom (y=5) to top (y=5*n-5)
    # Each leaf is spaced 10 units apart, starting at 5.
    def ivl_to_y(ivl):
        return {label: 5 + 10 * i for i, label in enumerate(ivl)}

    ivl_l = (ddata_l.get("ivl", order_l) if Z_l is not None else order_l)
    ivl_r = (ddata_r.get("ivl", order_r) if Z_r is not None else order_r)

    y_l = ivl_to_y(ivl_l)
    y_r = ivl_to_y(ivl_r)

    # Rank difference for line colour (concordant=blue, discordant=red)
    rank_l = {s: i for i, s in enumerate(ivl_l)}
    rank_r = {s: i for i, s in enumerate(ivl_r)}

    # ── draw connector lines ─────────────────────────────────────────────────
    ax_mid.set_xlim(0, 1)
    ax_mid.set_ylim(5 - 5, 10 * n)
    ax_mid.axis("off")

    max_diff = max(abs(rank_l[s] - rank_r[s]) for s in shared) or 1
    for s in shared:
        yl = y_l[s]
        yr = y_r[s]
        diff = abs(rank_l[s] - rank_r[s])
        alpha = 0.35 + 0.55 * (diff / max_diff)
        # concordant (same rank) → blue, discordant → red
        frac = diff / max_diff
        r_ = frac
        b_ = 1 - frac
        color = (r_, 0.2, b_, alpha)

        verts = [(0, yl), (0.5, yl), (0.5, yr), (1, yr)]
        codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
        patch = mpatches.PathPatch(Path(verts, codes),
                                   facecolor="none", edgecolor=color[:3],
                                   alpha=alpha, lw=1.3)
        ax_mid.add_patch(patch)

    # ── Spearman correlation of rank orders ───────────────────────────────────
    ranks_l_vec = [rank_l[s] for s in shared]
    ranks_r_vec = [rank_r[s] for s in shared]
    rho, pval = spearmanr(ranks_l_vec, ranks_r_vec)

    fig.suptitle(
        f"Tanglegram: {title_left}  ↔  {title_right}\n"
        f"Spearman ρ = {rho:.3f}  (p={pval:.3f})   |   "
        f"n shared genomes = {n}\n"
        f"Line colour: blue = concordant order, red = discordant",
        fontsize=10, y=1.00
    )

    return fig, rho, pval


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 – all 6 tanglegrams
# ══════════════════════════════════════════════════════════════════════════════
print("\nDrawing tanglegrams …")

pairs = list(combinations(REPLICONS, 2))
tangl_metrics = []

for rep_a, rep_b in pairs:
    Z_a, labels_a = linkages[rep_a]
    Z_b, labels_b = linkages[rep_b]
    title_a = f"{rep_a} {LABELS[rep_a]}"
    title_b = f"{rep_b} {LABELS[rep_b]}"

    result = tanglegram(
        Z_a, labels_a, Z_b, labels_b,
        title_a, title_b,
        COLORS[rep_a], COLORS[rep_b],
        genome_colors
    )
    if result is None:
        print(f"  Skipped {rep_a} vs {rep_b} (too few shared genomes)")
        continue

    fig_tg, rho, pval = result
    slug = f"{rep_a}_vs_{rep_b}"
    fig_tg.savefig(os.path.join(OUT_DIR, f"tanglegram_{slug}.pdf"),
                   bbox_inches="tight")
    fig_tg.savefig(os.path.join(OUT_DIR, f"tanglegram_{slug}.png"),
                   dpi=180, bbox_inches="tight")
    plt.close(fig_tg)
    tangl_metrics.append({"pair": f"{rep_a} vs {rep_b}",
                           "spearman_rho": round(rho, 4),
                           "pvalue": round(pval, 4)})
    print(f"  {rep_a:12s} vs {rep_b:12s}: ρ={rho:.3f}  p={pval:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 – 2×3 summary tanglegram panel
# ══════════════════════════════════════════════════════════════════════════════
print("\nDrawing summary panel …")
fig_sum, axes_sum = plt.subplots(2, 3, figsize=(22, 18))
axes_flat = axes_sum.flatten()

for ax_idx, (rep_a, rep_b) in enumerate(pairs):
    ax = axes_flat[ax_idx]
    mat_a = genome_matrices[rep_a]
    mat_b = genome_matrices[rep_b]
    shared = sorted(set(mat_a.index) & set(mat_b.index))
    if len(shared) < 3:
        ax.axis("off")
        continue

    mat_a_sh = mat_a.loc[shared]
    mat_b_sh = mat_b.loc[shared]
    Z_a, _ = compute_linkage(mat_a_sh)
    Z_b, _ = compute_linkage(mat_b_sh)
    if Z_a is None or Z_b is None:
        ax.axis("off")
        continue

    order_a = get_leaf_order(Z_a, shared)
    order_b = get_leaf_order(Z_b, shared)
    rank_a  = {s: i for i, s in enumerate(order_a)}
    rank_b  = {s: i for i, s in enumerate(order_b)}
    n = len(shared)

    # Draw heatmap-style matrix: rows=shared genomes in order_a, cols=order_b
    # Cell colour = same genome? (identity matrix permuted by order_b)
    perm_mat = np.zeros((n, n))
    for i, s in enumerate(order_a):
        j = order_b.index(s)
        perm_mat[i, j] = 1

    ax.imshow(perm_mat, cmap="Blues", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(order_b, rotation=90, fontsize=5, ha="right")
    ax.set_yticklabels(order_a, fontsize=5)
    ax.set_xlabel(f"{LABELS[rep_b]}", color=COLORS[rep_b], fontsize=8)
    ax.set_ylabel(f"{LABELS[rep_a]}", color=COLORS[rep_a], fontsize=8)

    rho, pval = spearmanr([rank_a[s] for s in shared],
                          [rank_b[s] for s in shared])
    ax.set_title(f"{rep_a} vs {rep_b}\nρ={rho:.3f}, p={pval:.3f}", fontsize=9)
    # Add diagonal reference line
    ax.plot([0, n-1], [0, n-1], "r--", lw=0.8, alpha=0.4)

fig_sum.suptitle(
    "Genome ordering concordance between replicons\n"
    "(blue dot = same genome; diagonal = perfect concordance)",
    fontsize=12, y=1.01)
plt.tight_layout()
fig_sum.savefig(os.path.join(OUT_DIR, "fig3_tanglegram_summary_panel.pdf"),
                bbox_inches="tight")
fig_sum.savefig(os.path.join(OUT_DIR, "fig3_tanglegram_summary_panel.png"),
                dpi=180, bbox_inches="tight")
plt.close()
print("  fig3 saved.")

# ── Save metrics ──────────────────────────────────────────────────────────────
pd.DataFrame(tangl_metrics).to_csv(
    os.path.join(OUT_DIR, "tanglegram_spearman_metrics.csv"), index=False)

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 – Motif contribution heatmap per replicon
# ══════════════════════════════════════════════════════════════════════════════
print("\nDrawing motif presence heatmaps per replicon (with dendrograms) …")

fig, axes = plt.subplots(2, 2, figsize=(22, 20))
axes_flat = axes.flatten()

for ax_idx, rep in enumerate(REPLICONS):
    ax = axes_flat[ax_idx]
    mat = genome_matrices[rep]
    Z, labels = compute_linkage(mat)
    if Z is None:
        ax.axis("off")
        continue
    order = get_leaf_order(Z, labels)
    mat_ord = mat.loc[order]

    # Only keep motifs present in ≥ 2 genomes (already filtered)
    # Further simplify for display: motifs in ≥ 10% of genomes
    min_prev = max(2, int(0.1 * len(mat)))
    motifs_show = mat.columns[mat.sum() >= min_prev].tolist()
    if not motifs_show:
        motifs_show = mat.columns.tolist()
    mat_show = mat_ord[motifs_show]

    im = ax.imshow(mat_show.values, aspect="auto", cmap="Blues",
                   vmin=0, vmax=1, interpolation="nearest")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=6)
    ax.set_xticks(range(len(motifs_show)))
    ax.set_xticklabels(motifs_show, rotation=60, ha="right", fontsize=6)
    ax.set_title(f"{LABELS[rep]} – motif presence\n"
                 f"(ordered by dendrogram, {len(motifs_show)} motifs shown)",
                 color=COLORS[rep], fontweight="bold", fontsize=9)
    # Colour y-tick labels
    for lbl in ax.get_yticklabels():
        lbl.set_color(genome_colors.get(lbl.get_text(), "black"))
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04,
                 label="Motif present (1=yes)")

plt.suptitle("Methylation motif presence per replicon "
             "(rows ordered by hierarchical clustering)", fontsize=12, y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig4_motif_heatmaps_per_replicon.pdf"),
            bbox_inches="tight")
fig.savefig(os.path.join(OUT_DIR, "fig4_motif_heatmaps_per_replicon.png"),
            dpi=180, bbox_inches="tight")
plt.close()
print("  fig4 saved.")

# ── Print summary ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
for m in tangl_metrics:
    print(f"  {m['pair']:28s}: ρ={m['spearman_rho']:+.3f}  p={m['pvalue']:.3f}")
print(f"\nOutputs in: {OUT_DIR}")
print("="*60)
