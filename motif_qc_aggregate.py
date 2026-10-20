#!/usr/bin/env python3
"""
Methylation motif QC — Phase 5: aggregate the 44 per-genome outputs of motif_sites_qc.py,
calibrate site-level "detected" calls, resolve shadow / weak motifs and build the
contig x family matrix. See methylation_motif_QC_methodology.md.

Steps
  1  Calibration: choose the GFF score cutoff t* at which site-level detected counts
     reproduce motifMaker nDetected (all sample x motif pairs); check nGenome vs enumerated sites
  2  Background: per genome, rate of GFF calls (score >= t) on A/C bases outside every
     enumerated motif site -> null for "methylated"
  3  GANTC footprint: kinetics at the N of GANTC (offset +1, same strand) by base, all genomes
  4  Final family status (retain / GANTC- or m6A-linked / weak / background)
  5  Contig x family matrix (fraction, n covered, presence) with NaN where a family was not
     scanned in that genome (motif not called there by motifMaker)

Outputs (default methyl_call/motif_QC/phase5/):
  calibration_by_threshold.tsv, calibration_pairs.tsv, background_by_genome.tsv,
  gantc_footprint_by_genome.tsv, footprint_profile.tsv, shadow_test_all.tsv,
  family_status.tsv, contig_family_long.tsv,
  contig_family_fraction.tsv / contig_family_presence.tsv          (retained families)
  contig_family_fraction_weak.tsv / contig_family_presence_weak.tsv (weak-flag families), figures/
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from Bio import SeqIO

BASE = "/n/projects/jp2992/Plasmidome_project_MOLNG_4509"

# ── Tunable thresholds (mirrored in the methodology doc) ─────────────────────
GRID            = np.arange(20, 121)   # candidate GFF score cutoffs
MIN_SITE_COV    = 25                   # same as motif_sites_qc.py
CONTIG_MIN_SITES = 10                  # covered sites needed to score a contig
CONTIG_PRESENT  = 0.50                 # calibrated fraction for presence on a contig
WEAK_MIN_FRAC   = 0.20                 # weak motif: genome-wide calibrated fraction
# Enrichment over background is reported but not used as a gate: at t* every motif that
# motifMaker reports is >100x above background, so it does not discriminate.
NEAR_M6A_SHARE  = 0.80                 # called sites near High m6A -> m6A-linked


def counts_at_or_above(scores, grid=GRID):
    """Number of scores >= each grid value."""
    s = np.sort(scores[~np.isnan(scores)])
    return len(s) - np.searchsorted(s, grid, side="left")


def read_gff_calls(path):
    g = pd.read_csv(path, sep="\t", comment="#", header=None, usecols=[0, 3, 5, 6, 8],
                    names=["contig", "pos", "score", "strand", "attr"],
                    dtype={"contig": str, "pos": np.int64, "score": np.float64, "strand": str, "attr": str})
    g["base"] = g["attr"].str.extract(r"context=([ACGTN]{41})", expand=False).str[20]
    return g.drop(columns="attr")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--qc-dir", default=f"{BASE}/methyl_call/motif_QC")
    ap.add_argument("--methyl-dir", default=f"{BASE}/methyl_call/ipdsummary_out")
    ap.add_argument("--flye-dir", default=f"{BASE}/flye_asm")
    ap.add_argument("--outdir", default=f"{BASE}/methyl_call/motif_QC/phase5")
    ap.add_argument("--sites-dir", default=None,
                    help="per-genome outputs of motif_sites_qc.py (default <qc-dir>/motif_sites; "
                         "use <qc-dir>/motif_sites_retained for --scan retained runs)")
    ap.add_argument("--tstar", type=int, default=None,
                    help="fix the GFF score cutoff instead of fitting it (e.g. 101 from the called-mode run)")
    args = ap.parse_args()
    sites_dir = args.sites_dir or f"{args.qc_dir}/motif_sites"
    os.makedirs(f"{args.outdir}/figures", exist_ok=True)

    calls = pd.read_csv(f"{args.qc_dir}/motif_calls_annotated.tsv", sep="\t")
    calls = calls[calls.sample_pass].copy()
    calls["motif_id"] = calls.motifString + "|" + calls.modificationType
    catalog = pd.read_csv(f"{args.qc_dir}/motif_catalog.tsv", sep="\t")
    fam_rep = dict(zip(catalog.family_id, catalog.representative))
    # motif_id -> family_id is the same in every genome (family = canonical key); global map so that
    # motifs scanned in genomes where motifMaker did not call them (--scan retained) keep their family
    fam_of = calls.drop_duplicates("motif_id").set_index("motif_id").family_id
    samples = [l.strip() for l in open(f"{args.qc_dir}/samples_pass.txt") if l.strip()]

    motif_hist, fam_contig_hist, bg_rows, contig_meta = [], [], [], []
    for i, s in enumerate(samples, 1):
        print(f"[{i}/{len(samples)}] {s}", flush=True)
        st = pd.read_csv(f"{sites_dir}/{s}.sites.tsv.gz", sep="\t",
                         usecols=["contig", "pos", "strand", "motif_id", "call_score",
                                  "coverage", "replicon"])
        st["family_id"] = st.motif_id.map(fam_of)

        # (1) per motif: all enumerated sites, detected counts across the score grid
        for mid, d in st.groupby("motif_id"):
            motif_hist.append(dict(sample=s, motif_id=mid, n_sites=len(d),
                                   counts=counts_at_or_above(d.call_score.values)))

        # (5) per family (union of variant sites) x contig: covered sites and detected counts
        fu = st.drop_duplicates(["family_id", "contig", "pos", "strand"])
        fu = fu[fu.coverage >= MIN_SITE_COV]
        for (fid, c), d in fu.groupby(["family_id", "contig"]):
            fam_contig_hist.append(dict(sample=s, family_id=fid, contig=c,
                                        replicon=d.replicon.iloc[0], n_covered=len(d),
                                        counts=counts_at_or_above(d.call_score.values)))

        # (2) background: GFF calls on A/C outside every enumerated motif site
        fasta = f"{args.flye_dir}/{s}_filtered/{s}_filtered_asm.fasta"
        lens = {r.id: len(r.seq) for r in SeqIO.parse(fasta, "fasta")}
        for c, L in lens.items():
            contig_meta.append(dict(sample=s, contig=c, length=L))
        g = read_gff_calls(f"{args.methyl_dir}/{s}/{s}_basemods.gff")
        keys = st[["contig", "pos", "strand"]].drop_duplicates()
        g = g.merge(keys.assign(in_motif=True), on=["contig", "pos", "strand"], how="left")
        out = g[g.in_motif.isna() & g.base.isin(["A", "C"])]
        # A/C positions over both strands = genome length; remove motif sites (all A or C)
        denom = sum(lens.values()) - len(keys)
        bg_rows.append(dict(sample=s, genome_len=sum(lens.values()), n_motif_sites=len(keys),
                            denom=denom, counts=counts_at_or_above(out.score.values)))

    # ── Step 1: calibration against motifMaker ──────────────────────────────
    mh = pd.DataFrame(motif_hist).merge(
        calls[["sample", "motif_id", "nDetected", "nGenome", "fraction", "tier", "family_id"]],
        on=["sample", "motif_id"], how="left")
    mh["family_id"] = mh.motif_id.map(fam_of)
    mh["called"] = mh.nDetected.notna()                      # False = scanned but not called in that genome
    C = np.vstack(mh.counts.values)                          # pairs x grid
    Cc, nd = C[mh.called.values], mh.nDetected[mh.called].values[:, None]
    logerr = np.abs(np.log((Cc + 1) / (nd + 1)))             # calibration uses called pairs only
    med_err = np.median(logerr, axis=0)
    k = int(np.where(GRID == args.tstar)[0][0]) if args.tstar else int(np.argmin(med_err))
    t_star = int(GRID[k])
    calib = pd.DataFrame({"threshold": GRID, "median_abs_log_ratio": med_err,
                          "median_ratio": np.median((Cc + 1) / (nd + 1), axis=0)})
    calib.to_csv(f"{args.outdir}/calibration_by_threshold.tsv", sep="\t", index=False)
    mh["n_detected_tstar"] = C[:, k]
    mh["site_fraction_tstar"] = mh.n_detected_tstar / mh.n_sites
    mh["sites_vs_nGenome"] = mh.n_sites / mh.nGenome
    mh.drop(columns="counts").to_csv(f"{args.outdir}/calibration_pairs.tsv", sep="\t", index=False)
    r = np.corrcoef(mh.fraction[mh.called], mh.site_fraction_tstar[mh.called])[0, 1]
    print(f"Calibration: t* = {t_star} (median |log ratio| {med_err[k]:.3f}); "
          f"site fraction vs motifMaker fraction r = {r:.3f}; "
          f"n_sites/nGenome median {mh.sites_vs_nGenome.median():.3f} "
          f"[{mh.sites_vs_nGenome.quantile(.05):.3f}, {mh.sites_vs_nGenome.quantile(.95):.3f}]")

    # ── Step 2: background ──────────────────────────────────────────────────
    bg = pd.DataFrame(bg_rows)
    B = np.vstack(bg.counts.values)
    bg["bg_calls_tstar"] = B[:, k]
    bg["bg_rate_tstar"] = bg.bg_calls_tstar / bg.denom
    bg["bg_rate_min_score"] = B[:, 0] / bg.denom
    bg.drop(columns="counts").to_csv(f"{args.outdir}/background_by_genome.tsv", sep="\t", index=False)
    bg_rate = dict(zip(bg["sample"], bg.bg_rate_tstar))
    print(f"Background at t*: median {bg.bg_rate_tstar.median():.2e} "
          f"[{bg.bg_rate_tstar.min():.2e}, {bg.bg_rate_tstar.max():.2e}]")

    # ── Step 3: footprints ──────────────────────────────────────────────────
    fp = pd.concat([pd.read_csv(f, sep="\t") for f in sorted(glob.glob(f"{sites_dir}/*.m6A_footprint.tsv"))],
                   ignore_index=True)
    prof = fp.groupby(["motif_id", "offset", "rel_strand", "base"]).agg(
        n_genomes=("sample", "nunique"), n_sites=("n", "sum"),
        median_of_genome_median_ipd=("median_ipd", "median"),
        mean_share_called_m4C=("share_called_m4C", "mean")).reset_index()
    prof.to_csv(f"{args.outdir}/footprint_profile.tsv", sep="\t", index=False)
    gfp = fp[(fp.motif_id == "GANTC|m6A") & (fp.offset == 1) & (fp.rel_strand == "same")]
    gw = gfp.pivot(index="sample", columns="base", values="median_ipd")
    gw.columns = [f"median_ipd_N={b}" for b in gw.columns]
    gm = gfp.pivot(index="sample", columns="base", values="share_called_m4C")
    gm.columns = [f"share_m4C_N={b}" for b in gm.columns]
    gfoot = gw.join(gm)
    gfoot["C_minus_maxAGT"] = gfoot["median_ipd_N=C"] - gfoot[["median_ipd_N=A", "median_ipd_N=G", "median_ipd_N=T"]].max(axis=1)
    gfoot.to_csv(f"{args.outdir}/gantc_footprint_by_genome.tsv", sep="\t")
    print("GANTC N-position median IPD (median over genomes): " +
          ", ".join(f"{c.split('=')[1]} {gfoot[c].median():.2f}" for c in gw.columns) +
          f"; C - max(A,G,T) > 0 in {(gfoot.C_minus_maxAGT > 0).sum()}/{len(gfoot)} genomes")

    # ── Step 4: family status ───────────────────────────────────────────────
    sh = pd.concat([pd.read_csv(f, sep="\t") for f in sorted(glob.glob(f"{sites_dir}/*.shadow_test.tsv"))],
                   ignore_index=True)
    sh.to_csv(f"{args.outdir}/shadow_test_all.tsv", sep="\t", index=False)
    mh["bg_rate"] = mh["sample"].map(bg_rate)
    mh["enrichment"] = mh.site_fraction_tstar / mh.bg_rate
    shf = sh.groupby("family_id").share_detected_near_m6A.median()
    rows = []
    for _, f in catalog.iterrows():
        d = mh[(mh.family_id == f.family_id) & mh.called]   # status from genomes where it was called
        rows.append(dict(
            family_id=f.family_id, representative=f.representative,
            modificationType=f.modificationType, phase2_status=f.phase2_status,
            n_genomes_called=d["sample"].nunique(),
            median_motifmaker_fraction=d.fraction.median(),
            median_site_fraction_tstar=d.site_fraction_tstar.median(),
            median_enrichment_over_bg=d.enrichment.median(),
            median_share_called_near_high_m6A=shf.get(f.family_id, np.nan)))
    fs = pd.DataFrame(rows)

    def final(r):
        if r.n_genomes_called == 0:            # family not scanned in this run (e.g. --scan retained)
            return "not_scanned"
        if r.phase2_status == "retain":
            return "retain"
        if r.phase2_status == "candidate_shadow":
            return "m6A_linked_excluded"
        if r.median_share_called_near_high_m6A >= NEAR_M6A_SHARE:
            return "m6A_linked_excluded"
        if r.median_site_fraction_tstar >= WEAK_MIN_FRAC:
            return "weak_methylation_flag"
        return "background_excluded"
    fs["final_status"] = fs.apply(final, axis=1)
    fs = fs.sort_values(["final_status", "n_genomes_called"], ascending=[True, False])
    fs.to_csv(f"{args.outdir}/family_status.tsv", sep="\t", index=False)
    print("Final family status:", fs.final_status.value_counts().to_dict())

    # ── Step 5: contig x family matrix ──────────────────────────────────────
    fc = pd.DataFrame(fam_contig_hist)
    fc["n_detected"] = np.vstack(fc.counts.values)[:, k]
    fc = fc.drop(columns="counts")
    fc["fraction"] = fc.n_detected / fc.n_covered
    fc["present"] = np.where(fc.n_covered >= CONTIG_MIN_SITES, (fc.fraction >= CONTIG_PRESENT).astype(float), np.nan)
    fc["representative"] = fc.family_id.map(fam_rep)
    fc = fc.merge(fs[["family_id", "final_status"]], on="family_id", how="left")
    cm = pd.DataFrame(contig_meta)
    fc = fc.merge(cm, on=["sample", "contig"], how="left")
    fc.to_csv(f"{args.outdir}/contig_family_long.tsv", sep="\t", index=False)

    col = lambda fid: f"{fid}_{fam_rep[fid]}"
    tags = pd.read_csv(f"{BASE}/contigs_to_tags.tsv", sep="\t", header=None, names=["contig", "replicon"])
    idx = cm.merge(tags, on="contig", how="left").fillna({"replicon": "Unassigned"}) \
            .set_index(["sample", "contig", "replicon", "length"])

    def wide(status, suffix):
        # NaN = family not scanned in that genome (not called by motifMaker) or < CONTIG_MIN_SITES covered sites
        sub = fc[fc.family_id.isin(fs[fs.final_status == status].family_id)].copy()
        sub["col"] = sub.family_id.map(col)
        out = {}
        for v in ("fraction", "present"):
            w = sub.pivot_table(index=["sample", "contig"], columns="col", values=v)
            out[v] = idx.reset_index().merge(w.reset_index(), on=["sample", "contig"], how="left") \
                        .set_index(idx.index.names)
        out["fraction"].to_csv(f"{args.outdir}/contig_family_fraction{suffix}.tsv", sep="\t")
        out["present"].to_csv(f"{args.outdir}/contig_family_presence{suffix}.tsv", sep="\t")
        print(f"Contig x family ({status}): {out['fraction'].shape[0]} contigs x "
              f"{out['fraction'].shape[1]} families ({int(np.isfinite(out['present'].values).sum())} scored cells)")
        return out["fraction"]

    frac = wide("retain", "")                         # main matrix: retained families only
    wide("weak_methylation_flag", "_weak")            # weak families kept apart

    # ── Figures ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].plot(calib.threshold, calib.median_abs_log_ratio, color="#2a6f97")
    ax[0].axvline(t_star, ls="--", color="grey")
    ax[0].set(xlabel="GFF score cutoff", ylabel="median |log((site count+1)/(nDetected+1))|",
              title=f"Calibration vs motifMaker (t* = {t_star})")
    tiercol = {"High": "#2a6f97", "Partial": "#e9a03b", "Low": "#b0b0b0"}
    mc = mh[mh.called]
    ax[1].scatter(mc.fraction, mc.site_fraction_tstar, c=mc.tier.map(tiercol), s=14, edgecolor="none")
    ax[1].plot([0, 1], [0, 1], ls=":", color="grey")
    ax[1].set(xlabel="motifMaker fraction", ylabel=f"site-level fraction (score ≥ {t_star})",
              title=f"Agreement (r = {r:.2f})")
    for _, b in bg.iterrows():
        ax[2].plot(GRID, b.counts / b.denom, color="#b0b0b0", lw=.6)
    ax[2].axvline(t_star, ls="--", color="grey")
    ax[2].set(yscale="log", xlabel="GFF score cutoff", ylabel="calls per A/C outside motifs",
              title="Background rate per genome")
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{args.outdir}/figures/fig3_calibration_background.png", dpi=200)
    fig.savefig(f"{args.outdir}/figures/fig3_calibration_background.pdf")

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    bases = ["A", "C", "G", "T"]
    for j, b in enumerate(bases):
        y = gfoot[f"median_ipd_N={b}"].values
        ax[0].scatter(np.full(len(y), j) + np.random.uniform(-.15, .15, len(y)), y, s=10,
                      color="#e9a03b" if b == "C" else "#2a6f97", edgecolor="none")
        y2 = gfoot[f"share_m4C_N={b}"].values
        ax[1].scatter(np.full(len(y2), j) + np.random.uniform(-.15, .15, len(y2)), y2, s=10,
                      color="#e9a03b" if b == "C" else "#2a6f97", edgecolor="none")
    ax[0].set(xticks=range(4), xticklabels=[f"GA{b}TC" for b in bases], ylabel="median IPD ratio at N",
              title="IPD at the N of GANTC (one dot per genome)")
    ax[1].set(xticks=range(4), xticklabels=[f"GA{b}TC" for b in bases], ylabel="share of N sites called m4C",
              title="m4C calls at the N of GANTC")
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{args.outdir}/figures/fig4_gantc_N_footprint.png", dpi=200)
    fig.savefig(f"{args.outdir}/figures/fig4_gantc_N_footprint.pdf")

    order = {"Chromosome": 0, "pSymB": 1, "pSymA": 2, "Other": 3, "Unassigned": 4}
    fr = frac.reset_index()
    fr = fr.assign(o=fr.replicon.map(order)).sort_values(["o", "sample", "length"], ascending=[True, True, False])
    M = fr[frac.columns].values.T
    fig, ax = plt.subplots(figsize=(14, max(4, .3 * len(frac.columns))))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#f2f2f2")
    im = ax.imshow(np.ma.masked_invalid(M), aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    ax.set_yticks(range(len(frac.columns)), frac.columns, fontsize=7)
    bounds = fr.o.values
    for x in np.where(np.diff(bounds) != 0)[0]:
        ax.axvline(x + .5, color="white", lw=1.5)
    ax.set_xticks([np.mean(np.where(bounds == v)[0]) for v in sorted(set(bounds))],
                  [k for k, v in sorted(order.items(), key=lambda kv: kv[1]) if v in set(bounds)])
    ax.set_title(f"Calibrated methylated fraction per contig (score ≥ {t_star}, ≥ {CONTIG_MIN_SITES} covered sites; grey = not scanned / too few sites)",
                 fontsize=9)
    fig.colorbar(im, ax=ax, fraction=.02)
    fig.tight_layout()
    fig.savefig(f"{args.outdir}/figures/fig5_contig_family_fraction.png", dpi=200)
    fig.savefig(f"{args.outdir}/figures/fig5_contig_family_fraction.pdf")
    print("done")


if __name__ == "__main__":
    main()
