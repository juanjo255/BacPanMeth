#!/usr/bin/env python3
"""
Dot plot of methylated motif families per replicon class.

  rows    motif families (retained; optionally also weak-flag families)
  columns Chromosome, pSymB, pSymA, accessory plasmids (contigs tagged "Other")
  size    number of genomes in which the motif is methylated on that replicon class
          (calibrated fraction >= --present on >= 1 contig with >= 10 covered sites)
  color   median calibrated methylated fraction across those genomes

Reads phase5/contig_family_long.tsv written by motif_qc_aggregate.py.
"""

import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

BASE = "/n/projects/jp2992/Plasmidome_project_MOLNG_4509"
COLS = [("Chromosome", "Chromosome"), ("pSymB", "pSymB"), ("pSymA", "pSymA"), ("Other", "Accessory\nplasmids")]
# Single-hue sequential ramp (blue 250 -> 700), light = lower fraction
SEQ = LinearSegmentedColormap.from_list(
    "blue_seq", ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase5-dir", default=f"{BASE}/methyl_call/motif_QC/phase5")
    ap.add_argument("--present", type=float, default=0.5, help="calibrated fraction for 'methylated'")
    ap.add_argument("--min-sites", type=int, default=10, help="covered sites needed to score a contig")
    ap.add_argument("--include-weak", action="store_true", help="add weak_methylation_flag families")
    ap.add_argument("--out", default=None, help="output prefix (default <phase5-dir>/figures/fig6_motif_dotplot)")
    args = ap.parse_args()
    out = args.out or f"{args.phase5_dir}/figures/fig6_motif_dotplot" + ("_with_weak" if args.include_weak else "")

    status = ["retain"] + (["weak_methylation_flag"] if args.include_weak else [])
    d = pd.read_csv(f"{args.phase5_dir}/contig_family_long.tsv", sep="\t")
    d = d[d.final_status.isin(status) & (d.n_covered >= args.min_sites)]

    # Per genome x family x replicon: methylated if any contig of that class passes; keep its best fraction
    g = d.groupby(["family_id", "representative", "final_status", "replicon", "sample"]).fraction.max().reset_index()
    g["methylated"] = g.fraction >= args.present
    m = g[g.methylated].groupby(["family_id", "representative", "final_status", "replicon"]).agg(
        n_genomes=("sample", "nunique"), median_fraction=("fraction", "median")).reset_index()
    tot = g[g.methylated].groupby("family_id")["sample"].nunique()

    # Row order: retained before weak, then by number of genomes, then name
    fams = m.drop_duplicates("family_id")[["family_id", "representative", "final_status"]].copy()
    fams["n_total"] = fams.family_id.map(tot)
    fams["weak"] = fams.final_status != "retain"
    fams = fams.sort_values(["weak", "n_total", "representative"], ascending=[True, False, True]).reset_index(drop=True)
    ypos = {f: i for i, f in enumerate(fams.family_id)}
    xpos = {r: j for j, (r, _) in enumerate(COLS)}
    m = m[m.replicon.isin(xpos)]

    nmax = int(m.n_genomes.max())
    size = lambda n: 18 + 420 * (np.asarray(n) / nmax)          # marker area, >= 8 px wide at n = 1
    vmin = min(0.5, m.median_fraction.min())

    fig_h = 1.2 + 0.32 * len(fams)
    fig, ax = plt.subplots(figsize=(6.4, fig_h))
    sc = ax.scatter(m.replicon.map(xpos), m.family_id.map(ypos), s=size(m.n_genomes),
                    c=m.median_fraction, cmap=SEQ, vmin=vmin, vmax=1,
                    edgecolor="white", linewidth=1.2, zorder=3)
    # Count label beside every dot (area alone is hard to read exactly); offset scales with dot radius
    for _, r in m.iterrows():
        ax.annotate(str(int(r.n_genomes)), (xpos[r.replicon], ypos[r.family_id]),
                    xytext=(4 + np.sqrt(size(r.n_genomes)) / 2, 0), textcoords="offset points",
                    va="center", fontsize=6.5, color="#555555")

    labels = [f"{r.representative}" + ("  (weak)" if r.weak else "") for r in fams.itertuples()]
    ax.set_yticks(range(len(fams)), labels, fontsize=7.5, family="monospace")
    ax.set_xticks(range(len(COLS)), [c[1] for c in COLS], fontsize=8.5)
    ax.set_xlim(-0.6, len(COLS) - 0.4)
    ax.set_ylim(len(fams) - 0.5, -0.5)
    if args.include_weak and fams.weak.any():
        ax.axhline(fams.weak.idxmax() - 0.5, color="#bbbbbb", lw=0.8, ls="--")
    ax.grid(axis="both", color="#eeeeee", lw=0.6, zorder=0)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(f"Methylated motif families across {d['sample'].nunique()} genomes", fontsize=10, loc="left")

    # Size legend (number of genomes) + colorbar (median fraction)
    ref = sorted({1, max(1, nmax // 4), max(1, nmax // 2), nmax})
    handles = [ax.scatter([], [], s=size(n), color="#9e9e9e", edgecolor="white") for n in ref]
    leg = ax.legend(handles, [str(n) for n in ref], title="Genomes", loc="upper left",
                    bbox_to_anchor=(1.02, 1.0), frameon=False, labelspacing=1.3, fontsize=7.5,
                    title_fontsize=8, borderpad=0.8)
    cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.25, anchor=(0, 0.0), shrink=0.45)
    cb.set_label("Median methylated\nfraction of sites", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    cb.outline.set_visible(False)

    fig.text(0.01, 0.005, f"Methylated = calibrated fraction ≥ {args.present} (GFF score > 100, ≥ {args.min_sites} sites "
             f"with coverage ≥ 25×) on ≥ 1 contig of that class.\nNo dot = no contig of that class with enough sites, "
             f"or the motif is not methylated there.", fontsize=6, color="#666666")
    fig.tight_layout()
    fig.savefig(out + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(out + ".pdf", bbox_inches="tight")
    m.to_csv(out + ".tsv", sep="\t", index=False)
    print(f"wrote {out}.png/.pdf/.tsv  ({len(fams)} families)")


if __name__ == "__main__":
    main()
