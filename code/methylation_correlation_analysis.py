#!/usr/bin/env python3
"""
Methylation pattern correlation analysis:
- Chromosome (>2.5 Mb) vs megaplasmids (800kb-2.5Mb) vs accessory plasmids (<800kb)
- Presence-absence matrix of methylated motifs per contig
- HDBSCAN clustering of accessory plasmids vs mobmess clusters
- Evaluation of accessory plasmid clustering with genome of origin
"""

import os
import re
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import hdbscan
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import LabelEncoder
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage
from collections import defaultdict

warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = "/n/projects/jp2992/Plasmidome_project_MOLNG_4509"
FLYE_DIR = os.path.join(BASE, "flye_asm")
METHYL_DIR = os.path.join(BASE, "methyl_call", "ipdsummary_out")
MOBMESS = os.path.join(BASE, "plasmids_decontamination", "mobmess_all_plsdb",
                        "mobmess_clusters_merged.txt")
OUT_DIR = os.path.join(BASE, "methyl_call", "methylation_correlation")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Size thresholds ───────────────────────────────────────────────────────────
CHR_MIN    = 2_500_000   # > 2.5 Mb → chromosome
MEGA_MIN   =   800_000   # 800 kb – 2.5 Mb → megaplasmid
# < 800 kb → accessory plasmid

def classify_contig(length):
    if length >= CHR_MIN:
        return "chromosome"
    elif length >= MEGA_MIN:
        return "megaplasmid"
    else:
        return "accessory_plasmid"

# ── Step 1: collect contig lengths from assembly_info.txt ─────────────────────
print("Step 1: Collecting contig lengths …")
contig_meta = {}   # full_contig_name → {length, category, sample}

asm_dirs = sorted(glob.glob(os.path.join(FLYE_DIR, "*_filtered")))
for asm_dir in asm_dirs:
    sample = os.path.basename(asm_dir).replace("_filtered", "")
    asm_info = os.path.join(asm_dir, "assembly_info.txt")
    if not os.path.exists(asm_info):
        continue
    df = pd.read_csv(asm_info, sep="\t", comment="#",
                     names=["seq_name","length","cov","circ","repeat",
                            "mult","alt_group","graph_path"])
    for _, row in df.iterrows():
        full_name = f"{row.seq_name}_{sample}_filtered"
        contig_meta[full_name] = {
            "length": int(row.length),
            "category": classify_contig(int(row.length)),
            "sample": sample,
        }

print(f"  {len(contig_meta)} contigs from {len(asm_dirs)} assemblies")
cat_counts = pd.Series([v["category"] for v in contig_meta.values()]).value_counts()
print(cat_counts.to_string())

# ── Step 2: parse basemods_w_motif.gff → per-contig motif counts ──────────────
print("\nStep 2: Parsing methylation GFF files …")

# motif=X or motif=X,Y (multiple motifs on same position)
motif_re = re.compile(r"motif=([^;]+)")

contig_motif_counts = defaultdict(lambda: defaultdict(int))   # {contig: {motif: count}}

methyl_samples = sorted(os.listdir(METHYL_DIR))
for msample in methyl_samples:
    gff_path = os.path.join(METHYL_DIR, msample, f"{msample}_basemods_w_motif.gff")
    if not os.path.exists(gff_path):
        continue
    with open(gff_path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.rstrip().split("\t")
            if len(parts) < 9:
                continue
            contig = parts[0]
            attrs  = parts[8]
            m = motif_re.search(attrs)
            if not m:
                continue
            for motif in m.group(1).split(","):
                motif = motif.strip()
                if motif:
                    contig_motif_counts[contig][motif] += 1

print(f"  Motif data for {len(contig_motif_counts)} contigs")

# ── Step 3: build presence-absence matrix ─────────────────────────────────────
print("\nStep 3: Building presence-absence matrix …")

# Only keep contigs that exist in both contig_meta and motif data
contigs_with_both = [c for c in contig_meta if c in contig_motif_counts]
print(f"  Contigs with both assembly info and motif data: {len(contigs_with_both)}")

# Discover all unique motifs (single motifs only – not combined strings)
all_motifs = sorted({m for c in contigs_with_both
                     for m in contig_motif_counts[c]})
print(f"  Unique motifs: {all_motifs}")

rows = []
for contig in contigs_with_both:
    meta = contig_meta[contig]
    row = {"contig": contig, "sample": meta["sample"],
           "length": meta["length"], "category": meta["category"]}
    for motif in all_motifs:
        row[motif] = 1 if contig_motif_counts[contig].get(motif, 0) > 0 else 0
        row[f"{motif}_density"] = (contig_motif_counts[contig].get(motif, 0)
                                   / meta["length"] * 1000)
    rows.append(row)

df_full = pd.DataFrame(rows).set_index("contig")
df_full.to_csv(os.path.join(OUT_DIR, "contig_motif_matrix.csv"))
print(f"  Matrix saved: {df_full.shape}")

# ── Step 4: load mobmess clusters ────────────────────────────────────────────
print("\nStep 4: Loading mobmess clusters …")
mob_cols = ["contig", "cluster_id", "type", "circular", "representative",
            "cluster_name", "dataset", "contig_dup", "length_mob", "gc"]
mob_df = pd.read_csv(MOBMESS, sep="\t", header=None, names=mob_cols)
mob_map = dict(zip(mob_df["contig"], mob_df["cluster_id"]))
mob_name_map = dict(zip(mob_df["contig"], mob_df["cluster_name"]))
print(f"  {len(mob_df)} entries, {mob_df['cluster_id'].nunique()} clusters")

# ── Step 5: HDBSCAN clustering on accessory plasmids ─────────────────────────
print("\nStep 5: HDBSCAN clustering of accessory plasmids …")

acc = df_full[df_full["category"] == "accessory_plasmid"].copy()
print(f"  Accessory plasmids: {len(acc)}")

feat_cols = all_motifs  # binary presence columns
X_acc = acc[feat_cols].values.astype(float)

# PCA for visualization (Jaccard-like analysis on binary data)
pca = PCA(n_components=min(10, X_acc.shape[1], X_acc.shape[0]-1))
X_pca = pca.fit_transform(X_acc)
print(f"  PCA variance explained (first 5): "
      f"{pca.explained_variance_ratio_[:5].round(3)}")

# HDBSCAN on PCA-reduced data
clusterer = hdbscan.HDBSCAN(min_cluster_size=3, min_samples=2,
                             metric='euclidean', cluster_selection_method='eom')
hdb_labels = clusterer.fit_predict(X_pca)
acc = acc.copy()
acc["hdbscan_cluster"] = hdb_labels
n_clusters = len(set(hdb_labels)) - (1 if -1 in hdb_labels else 0)
n_noise = (hdb_labels == -1).sum()
print(f"  HDBSCAN: {n_clusters} clusters, {n_noise} noise points")

# Map mobmess cluster to each accessory plasmid
acc["mobmess_cluster"] = acc.index.map(mob_map)
acc["mobmess_name"]    = acc.index.map(mob_name_map)

# ── 2D embedding for visualization
if X_pca.shape[0] > 5:
    perp = min(30, X_pca.shape[0] - 1)
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42)
    X_2d = tsne.fit_transform(X_pca)
else:
    X_2d = X_pca[:, :2]
acc["tsne_1"] = X_2d[:, 0]
acc["tsne_2"] = X_2d[:, 1]

# ── Step 6: compare HDBSCAN clusters vs mobmess clusters ─────────────────────
print("\nStep 6: Comparing HDBSCAN vs mobmess clusters …")

acc_labeled = acc.dropna(subset=["mobmess_cluster"]).copy()
acc_labeled = acc_labeled[acc_labeled["hdbscan_cluster"] != -1].copy()

if len(acc_labeled) > 10:
    ari = adjusted_rand_score(acc_labeled["mobmess_cluster"],
                              acc_labeled["hdbscan_cluster"])
    nmi = normalized_mutual_info_score(acc_labeled["mobmess_cluster"],
                                       acc_labeled["hdbscan_cluster"])
    print(f"  ARI (HDBSCAN vs mobmess): {ari:.4f}")
    print(f"  NMI (HDBSCAN vs mobmess): {nmi:.4f}")
else:
    ari, nmi = np.nan, np.nan
    print("  Not enough labeled data for ARI/NMI")

# ── Step 7: evaluate if accessory plasmids cluster with their genome ───────────
print("\nStep 7: Genome-of-origin clustering analysis …")

# Include all sequence types in a joint PCA
X_all = df_full[feat_cols].values.astype(float)
pca_all = PCA(n_components=min(10, X_all.shape[1], X_all.shape[0]-1))
X_all_pca = pca_all.fit_transform(X_all)

if X_all_pca.shape[0] > 5:
    perp2 = min(30, X_all_pca.shape[0] - 1)
    tsne2 = TSNE(n_components=2, perplexity=perp2, random_state=42)
    X_all_2d = tsne2.fit_transform(X_all_pca)
else:
    X_all_2d = X_all_pca[:, :2]

df_full["tsne_1"] = X_all_2d[:, 0]
df_full["tsne_2"] = X_all_2d[:, 1]

# ── Step 8: save results ──────────────────────────────────────────────────────
acc.to_csv(os.path.join(OUT_DIR, "accessory_plasmid_clusters.csv"))
df_full.to_csv(os.path.join(OUT_DIR, "all_contigs_tsne.csv"))

metrics = pd.DataFrame([{"metric": "ARI_hdbscan_vs_mobmess", "value": ari},
                         {"metric": "NMI_hdbscan_vs_mobmess", "value": nmi},
                         {"metric": "n_hdbscan_clusters", "value": n_clusters},
                         {"metric": "n_noise_points", "value": n_noise},
                         {"metric": "n_accessory_plasmids", "value": len(acc)}])
metrics.to_csv(os.path.join(OUT_DIR, "clustering_metrics.csv"), index=False)
print("  Tables saved.")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ═══════════════════════════════════════════════════════════════════════════════

# ── Fig 1: Presence-absence heatmap ───────────────────────────────────────────
print("\nGenerating figures …")

fig, axes = plt.subplots(1, 3, figsize=(16, max(6, len(all_motifs))))
cat_order = ["chromosome", "megaplasmid", "accessory_plasmid"]
cat_labels = {"chromosome": "Chromosome", "megaplasmid": "Megaplasmid",
              "accessory_plasmid": "Accessory plasmid"}
cat_colors = {"chromosome": "#e74c3c", "megaplasmid": "#3498db",
              "accessory_plasmid": "#2ecc71"}

for ax, cat in zip(axes, cat_order):
    sub = df_full[df_full["category"] == cat][feat_cols]
    if sub.empty:
        ax.set_title(cat_labels[cat] + " (no data)")
        continue
    # Mean fraction present across contigs
    means = sub.mean().rename("fraction_present")
    ax.barh(means.index, means.values, color=cat_colors[cat], edgecolor="white")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Fraction of contigs with motif")
    ax.set_title(f"{cat_labels[cat]}\n(n={len(sub)})")
    ax.axvline(0.5, color="grey", lw=0.8, ls="--")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig1_motif_presence_by_category.pdf"),
            bbox_inches="tight")
plt.savefig(os.path.join(OUT_DIR, "fig1_motif_presence_by_category.png"),
            dpi=150, bbox_inches="tight")
plt.close()

# ── Fig 2: Full presence-absence heatmap (clustered) ──────────────────────────
fig, ax = plt.subplots(figsize=(max(8, len(all_motifs) * 1.2),
                                max(8, len(df_full) * 0.15)))
# Sort rows by category then sample
df_heatmap = df_full.sort_values(["category", "sample"])[feat_cols]
row_colors_list = [cat_colors[df_full.loc[c, "category"]] for c in df_heatmap.index]
row_colors_ser = pd.Series(row_colors_list, index=df_heatmap.index, name="Category")
g = sns.clustermap(df_heatmap, cmap="Blues", figsize=(12, max(10, len(df_heatmap)*0.18)),
                   row_colors=row_colors_ser, col_cluster=False, row_cluster=True,
                   yticklabels=False, linewidths=0, cbar_kws={"label": "Motif present"})
handles = [mpatches.Patch(color=c, label=cat_labels[k])
           for k, c in cat_colors.items()]
g.ax_heatmap.legend(handles=handles, bbox_to_anchor=(1.15, 1), loc="upper left",
                    frameon=True, title="Category")
g.fig.suptitle("Methylation motif presence-absence across all contigs", y=1.01)
g.savefig(os.path.join(OUT_DIR, "fig2_heatmap_all_contigs.pdf"), bbox_inches="tight")
g.savefig(os.path.join(OUT_DIR, "fig2_heatmap_all_contigs.png"), dpi=150,
          bbox_inches="tight")
plt.close()

# ── Fig 3: HDBSCAN clusters on accessory plasmids (t-SNE) ─────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 3a: coloured by HDBSCAN cluster
unique_hdb = sorted(set(acc["hdbscan_cluster"]))
hdb_palette = plt.cm.tab20(np.linspace(0, 1, max(len(unique_hdb), 2)))
hdb_colors = {c: ("lightgrey" if c == -1 else hdb_palette[i])
              for i, c in enumerate(unique_hdb)}
ax = axes[0]
for clust, grp in acc.groupby("hdbscan_cluster"):
    ax.scatter(grp["tsne_1"], grp["tsne_2"], c=[hdb_colors[clust]],
               label=("Noise" if clust == -1 else f"HDB {clust}"),
               s=60, alpha=0.8, edgecolors="white", lw=0.5)
ax.set_title(f"Accessory plasmids – HDBSCAN clusters\n"
             f"({n_clusters} clusters, ARI vs mobmess={ari:.3f})")
ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7,
          title="HDB cluster", ncol=2)

# 3b: coloured by mobmess cluster name
ax = axes[1]
mob_names = acc["mobmess_name"].fillna("Not in mobmess")
unique_mob = sorted(mob_names.unique())
mob_palette = plt.cm.tab20(np.linspace(0, 1, max(len(unique_mob), 2)))
mob_color_map = {n: mob_palette[i] for i, n in enumerate(unique_mob)}
for name, grp in acc.groupby(mob_names):
    color = mob_color_map[name]
    ax.scatter(grp["tsne_1"], grp["tsne_2"], c=[color], label=name,
               s=60, alpha=0.8, edgecolors="white", lw=0.5)
ax.set_title("Accessory plasmids – mobmess clusters")
ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7,
          title="Mobmess cluster", ncol=1)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig3_accessory_hdbscan_vs_mobmess.pdf"),
            bbox_inches="tight")
plt.savefig(os.path.join(OUT_DIR, "fig3_accessory_hdbscan_vs_mobmess.png"),
            dpi=150, bbox_inches="tight")
plt.close()

# ── Fig 4: Genome-of-origin analysis (all contigs t-SNE) ─────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# 4a: coloured by category
ax = axes[0]
for cat, grp in df_full.groupby("category"):
    ax.scatter(grp["tsne_1"], grp["tsne_2"], c=cat_colors[cat],
               label=cat_labels[cat], s=60, alpha=0.8, edgecolors="white", lw=0.5)
ax.set_title("All contigs – sequence category")
ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
ax.legend(title="Category")

# 4b: coloured by sample (genome), shape by category
ax = axes[1]
samples = sorted(df_full["sample"].unique())
sample_palette = plt.cm.tab20(np.linspace(0, 1, max(len(samples), 2)))
sample_color_map = {s: sample_palette[i] for i, s in enumerate(samples)}
markers = {"chromosome": "^", "megaplasmid": "s", "accessory_plasmid": "o"}
legend_patches = []
for cat, cat_grp in df_full.groupby("category"):
    for samp, grp in cat_grp.groupby("sample"):
        ax.scatter(grp["tsne_1"], grp["tsne_2"],
                   c=[sample_color_map[samp]],
                   marker=markers[cat],
                   s=(120 if cat == "chromosome" else 80 if cat == "megaplasmid" else 50),
                   alpha=0.85, edgecolors="white", lw=0.3)

# Legend: markers for category
for cat, mk in markers.items():
    legend_patches.append(plt.Line2D([0], [0], marker=mk, color="grey",
                                     linestyle="None", markersize=8,
                                     label=cat_labels[cat]))
ax.legend(handles=legend_patches, title="Shape = category", loc="upper right")
ax.set_title("All contigs – genome of origin\n(color=sample, shape=category)")
ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig4_genome_origin_tsne.pdf"), bbox_inches="tight")
plt.savefig(os.path.join(OUT_DIR, "fig4_genome_origin_tsne.png"), dpi=150,
            bbox_inches="tight")
plt.close()

# ── Fig 5: Per-sample heatmap showing motif presence across contig types ───────
# For each sample show a small heatmap: rows=contigs (grouped by type), cols=motifs
samples_with_data = df_full["sample"].unique()
n_samp = len(samples_with_data)
fig, axes = plt.subplots(n_samp, 1, figsize=(max(8, len(all_motifs) * 0.8),
                                              n_samp * 0.8 + 2))
if n_samp == 1:
    axes = [axes]

for ax, samp in zip(axes, sorted(samples_with_data)):
    sub = df_full[df_full["sample"] == samp].sort_values("category")
    mat = sub[feat_cols]
    if mat.empty:
        ax.set_visible(False)
        continue
    im = ax.imshow(mat.values, aspect="auto", cmap="Blues",
                   vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks(range(len(all_motifs)))
    ax.set_xticklabels(all_motifs, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(sub)))
    # Label rows by category abbreviation
    row_labels = [f"{r['category'][0].upper()}{i}"
                  for i, (_, r) in enumerate(sub.iterrows())]
    ax.set_yticklabels(row_labels, fontsize=6)
    ax.set_ylabel(samp, fontsize=7, rotation=0, ha="right", labelpad=60)

plt.suptitle("Per-sample motif presence (C=chromosome, M=megaplasmid, A=accessory)",
             y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig5_per_sample_motif_presence.pdf"),
            bbox_inches="tight")
plt.savefig(os.path.join(OUT_DIR, "fig5_per_sample_motif_presence.png"),
            dpi=150, bbox_inches="tight")
plt.close()

# ── Fig 6: Contingency heatmap – HDBSCAN vs mobmess ──────────────────────────
if not acc_labeled.empty and len(acc_labeled) > 5:
    cross = pd.crosstab(acc_labeled["hdbscan_cluster"],
                        acc_labeled["mobmess_cluster"])
    fig, ax = plt.subplots(figsize=(max(6, cross.shape[1] * 0.5),
                                   max(4, cross.shape[0] * 0.4)))
    sns.heatmap(cross, annot=True, fmt="d", cmap="YlOrRd", ax=ax,
                linewidths=0.5, cbar_kws={"label": "# plasmids"})
    ax.set_xlabel("Mobmess cluster ID")
    ax.set_ylabel("HDBSCAN cluster")
    ax.set_title(f"HDBSCAN vs Mobmess clusters\nARI={ari:.3f}  NMI={nmi:.3f}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig6_hdbscan_vs_mobmess_contingency.pdf"),
                bbox_inches="tight")
    plt.savefig(os.path.join(OUT_DIR, "fig6_hdbscan_vs_mobmess_contingency.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

# ── Summary print ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Total contigs analysed : {len(df_full)}")
print(f"  Chromosomes          : {(df_full.category=='chromosome').sum()}")
print(f"  Megaplasmids         : {(df_full.category=='megaplasmid').sum()}")
print(f"  Accessory plasmids   : {(df_full.category=='accessory_plasmid').sum()}")
print(f"\nMotifs detected        : {all_motifs}")
print(f"\nHDBSCAN (accessory)    : {n_clusters} clusters, {n_noise} noise")
print(f"ARI vs mobmess         : {ari:.4f}")
print(f"NMI vs mobmess         : {nmi:.4f}")
print(f"\nOutputs in: {OUT_DIR}")
print("="*60)
