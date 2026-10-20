#!/usr/bin/env python3
"""
Methylation motif QC & harmonization — Phases 0–2 (motif.csv level only).
See methylation_motif_QC_methodology.md for the rationale of each rule.

Phase 0  sample-level gate (final_dataset membership, assembly present, GANTC control, coverage)
Phase 1  per-call confidence tiers (High / Partial / Low)
Phase 2  sequence-level harmonization:
           - N-trimming and 0-based modified position (centerPos is 1-based here)
           - reverse-complement canonical key (merges per-strand reports of one motif)
           - families: motifs whose modified base aligns and whose IUPAC intersection
             keeps >= 50% of the sites of both motifs (no chaining via degenerate motifs),
             plus strict subsumption between High-tier motifs (joins halves of one system)
           - candidate m6A "shadows": motifs that always contain a High-tier m6A
             motif with the modified base within +-5 bp of the m6A base

Outputs (methyl_call/motif_QC/):
  sample_qc.tsv, motif_calls_annotated.tsv, motif_catalog.tsv,
  genome_family_matrix.tsv, figures/
"""

import argparse
import math
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/n/projects/jp2992/Plasmidome_project_MOLNG_4509"

# ── Tunable thresholds (mirrored in the methodology doc) ─────────────────────
GANTC_MIN_FRAC   = 0.90   # Phase 0 positive control
MIN_MEDIAN_COV   = 50     # Phase 0 coverage gate
HIGH_FRAC        = 0.85   # Phase 1 tiers
PARTIAL_FRAC     = 0.40
MIN_NGENOME      = 50
MAX_MODBASE_LEN  = 15     # long 'modified_base' motifs are motifMaker over-fits
FAMILY_MAX_BITS  = 1.0    # Phase 2: intersection may cost at most 1 bit vs EACH motif (>=50% mutual site overlap)
SHADOW_MAX_DIST  = 5      # Phase 2: candidate shadow window (bp)

# ── IUPAC helpers ─────────────────────────────────────────────────────────────
IUPAC = {
    "A": "A", "C": "C", "G": "G", "T": "T",
    "R": "AG", "Y": "CT", "S": "CG", "W": "AT", "K": "GT", "M": "AC",
    "B": "CGT", "D": "AGT", "H": "ACT", "V": "ACG", "N": "ACGT",
}
SET2CODE = {frozenset(v): k for k, v in IUPAC.items()}
COMP = {"A": "T", "C": "G", "G": "C", "T": "A", "R": "Y", "Y": "R", "S": "S", "W": "W",
        "K": "M", "M": "K", "B": "V", "V": "B", "D": "H", "H": "D", "N": "N"}


def revcomp(m):
    return "".join(COMP[c] for c in reversed(m))


def bits(m):
    """Information content of an IUPAC motif (log2 of inverse expected site frequency)."""
    return sum(math.log2(4 / len(IUPAC[c])) for c in m)


def trim_n(motif, pos0):
    """Strip flanking N and shift the modified-base index accordingly."""
    left = len(motif) - len(motif.lstrip("N"))
    core = motif.strip("N")
    return core, pos0 - left


def canonical(motif, pos0):
    """Orientation-independent key: lexicographically smaller of motif / revcomp.
    Returns (key, pos0 in key orientation, orientation flag)."""
    rc = revcomp(motif)
    if rc < motif:
        return rc, len(motif) - 1 - pos0, "rc"
    return motif, pos0, "fw"


def align_on_modbase(a, pa, b, pb):
    """Pad two motifs with N so their modified bases share an index."""
    left = max(pa, pb)
    a2 = "N" * (left - pa) + a
    b2 = "N" * (left - pb) + b
    L = max(len(a2), len(b2))
    return a2.ljust(L, "N"), b2.ljust(L, "N")


def intersect(a, b):
    """Position-wise IUPAC intersection; None if any position is empty."""
    out = []
    for x, y in zip(a, b):
        s = set(IUPAC[x]) & set(IUPAC[y])
        if not s:
            return None
        out.append(SET2CODE[frozenset(s)])
    return "".join(out)


def same_family(a, pa, b, pb):
    """True if, with modified bases aligned (trying both orientations of b), the IUPAC
    intersection keeps >= 2^-FAMILY_MAX_BITS of the sites of BOTH motifs. Using the less
    specific motif as reference stops degenerate motifs (e.g. SVVCNK) acting as hubs."""
    for bb, pbb in ((b, pb), (revcomp(b), len(b) - 1 - pb)):
        a2, b2 = align_on_modbase(a, pa, bb, pbb)
        inter = intersect(a2, b2)
        if inter is None:
            continue
        if bits(inter) - min(bits(a), bits(bb)) <= FAMILY_MAX_BITS:
            return True
    return False


def is_subsumed(a, pa, b, pb):
    """True if, with modified bases aligned (either orientation of b), every site of one
    motif is also a site of the other (IUPAC subset). Only applied to High-tier units, so it
    joins halves of one system (e.g. BCTCGAGV and GACTCGAG under CTCGAG) without letting
    degenerate low-confidence motifs act as hubs."""
    for bb, pbb in ((b, pb), (revcomp(b), len(b) - 1 - pb)):
        a2, b2 = align_on_modbase(a, pa, bb, pbb)
        inter = intersect(a2, b2)
        if inter is not None and abs(bits(inter) - max(bits(a), bits(bb))) < 1e-9:
            return True
    return False


def contains_at(a, m):
    """Offsets o where every site of `a` is guaranteed to contain `m` starting at a[o]
    (a[o+i] ⊆ m[i] for all i, m fully inside a)."""
    hits = []
    for o in range(0, len(a) - len(m) + 1):
        if all(set(IUPAC[a[o + i]]) <= set(IUPAC[m[i]]) for i in range(len(m))):
            hits.append(o)
    return hits


def shadow_candidates(a, pa, m6a_list):
    """Return (m6A motif, signed offset) pairs where `a` always contains a High m6A motif
    whose methylated base lies within SHADOW_MAX_DIST of a's modified base."""
    out = []
    for m, pm in m6a_list:
        if m == a:
            continue
        for mm, pmm in ((m, pm), (revcomp(m), len(m) - 1 - pm)):
            for o in contains_at(a, mm):
                d = pa - (o + pmm)
                if 0 < abs(d) <= SHADOW_MAX_DIST:
                    out.append((m, d))
    return sorted(set(out))


# ── Union-find for families ──────────────────────────────────────────────────
class UF:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--methyl-dir", default=f"{BASE}/methyl_call/ipdsummary_out")
    ap.add_argument("--final-dataset", default=f"{BASE}/final_dataset.txt")
    ap.add_argument("--flye-dir", default=f"{BASE}/flye_asm")
    ap.add_argument("--outdir", default=f"{BASE}/methyl_call/motif_QC")
    args = ap.parse_args()
    os.makedirs(f"{args.outdir}/figures", exist_ok=True)

    # ── Load all motif.csv ───────────────────────────────────────────────────
    frames = []
    for s in sorted(os.listdir(args.methyl_dir)):
        f = f"{args.methyl_dir}/{s}/{s}_motif.csv"
        if os.path.exists(f):
            frames.append(pd.read_csv(f).assign(sample=s))
    calls = pd.concat(frames, ignore_index=True)
    calls["pos0_raw"] = calls["centerPos"] - 1           # centerPos is 1-based in these files
    print(f"Loaded {len(calls)} motif calls from {calls['sample'].nunique()} samples")

    # ── Phase 0: sample gate ─────────────────────────────────────────────────
    final = {l.strip().removesuffix("_filtered") for l in open(args.final_dataset) if l.strip()}
    rows = []
    for s, d in calls.groupby("sample"):
        g = d[(d.motifString == "GANTC") & (d.modificationType == "m6A")]
        gfrac = g.fraction.max() if len(g) else np.nan
        r = dict(sample=s,
                 in_final_dataset=s in final,
                 assembly_present=os.path.exists(f"{args.flye_dir}/{s}_filtered/{s}_filtered_asm.fasta"),
                 GANTC_fraction=gfrac,
                 median_coverage=d.meanCoverage.median(),
                 n_motif_calls=len(d))
        reasons = []
        if not r["in_final_dataset"]:
            reasons.append("not_in_final_dataset")
        if not r["assembly_present"]:
            reasons.append("no_assembly")
        if not (gfrac >= GANTC_MIN_FRAC):
            reasons.append("GANTC_control_failed")
        if r["median_coverage"] < MIN_MEDIAN_COV:
            reasons.append("low_coverage")
        r["pass"] = not reasons
        r["fail_reasons"] = ";".join(reasons)
        rows.append(r)
    sample_qc = pd.DataFrame(rows)
    sample_qc.to_csv(f"{args.outdir}/sample_qc.tsv", sep="\t", index=False)
    keep = set(sample_qc.loc[sample_qc["pass"], "sample"])
    print(f"Phase 0: {len(keep)}/{len(sample_qc)} samples pass;",
          sample_qc.loc[~sample_qc["pass"], ["sample", "fail_reasons"]].to_dict("records"))

    # ── Phase 1: tiers ───────────────────────────────────────────────────────
    def tier(r):
        if r.nGenome < MIN_NGENOME:
            return "Low"
        if r.modificationType == "modified_base" and len(r.motifString) > MAX_MODBASE_LEN:
            return "Low"
        if r.fraction >= HIGH_FRAC:
            return "High"
        if r.fraction >= PARTIAL_FRAC:
            return "Partial"
        return "Low"
    calls["tier"] = calls.apply(tier, axis=1)
    calls["sample_pass"] = calls["sample"].isin(keep)

    # ── Phase 2a: normalize + canonical key ─────────────────────────────────
    norm = calls.apply(lambda r: trim_n(r.motifString, r.pos0_raw), axis=1)
    calls["motif_trim"] = [n[0] for n in norm]
    calls["pos0"] = [n[1] for n in norm]
    can = calls.apply(lambda r: canonical(r.motif_trim, r.pos0), axis=1)
    calls["canon_key"] = [c[0] for c in can]
    calls["canon_pos0"] = [c[1] for c in can]
    calls["canon_orient"] = [c[2] for c in can]
    calls["is_palindrome"] = calls.motif_trim == calls.motif_trim.map(revcomp)

    # Per-strand partner within a genome = another call with the same canonical key
    partner = {}
    for (s, k), d in calls.groupby(["sample", "canon_key"]):
        for i in d.index:
            others = d.index.difference([i])
            partner[i] = ",".join(calls.loc[others, "motifString"]) if len(others) else ""
    calls["rc_partner_in_genome"] = calls.index.map(partner)

    # ── Phase 2b: families over (canonical key, modified position, mod type) ─
    # One unit per canonical key: both strands of a bipartite/non-palindromic motif share a key,
    # whatever their modified position or reported mod type. Units are compared on every pair
    # of their modified positions.
    unit_pos = calls[calls.sample_pass].groupby("canon_key").canon_pos0.apply(lambda x: sorted(set(x)))
    units = list(unit_pos.index)
    # Units called High (same mod type) in >= 1 pass genome; eligible for subsumption merging
    high_units = set(calls[calls.sample_pass & (calls.tier == "High")].canon_key)
    uf = UF()
    for u in units:
        uf.find(u)
    for i in range(len(units)):
        for j in range(i + 1, len(units)):
            a, b = units[i], units[j]
            pairs = [(pa, pb) for pa in unit_pos[a] for pb in unit_pos[b]]
            if any(same_family(a, pa, b, pb) for pa, pb in pairs):
                uf.union(a, b)
            elif a in high_units and b in high_units and \
                    any(is_subsumed(a, pa, b, pb) for pa, pb in pairs):
                uf.union(a, b)
    groups = defaultdict(list)
    for u in units:
        groups[uf.find(u)].append(u)

    # ── Phase 2c: candidate shadows against High-tier m6A motifs ────────────
    high_m6a = calls[(calls.sample_pass) & (calls.tier == "High") & (calls.modificationType == "m6A")]
    m6a_by_sample = {s: list(zip(d.motif_trim, d.pos0)) for s, d in high_m6a.groupby("sample")}
    sh = []
    for r in calls.itertuples():
        if r.modificationType == "m6A" and r.tier == "High":
            sh.append("")
            continue
        cands = shadow_candidates(r.motif_trim, r.pos0, m6a_by_sample.get(r.sample, []))
        sh.append(";".join(f"{m}@{d:+d}" for m, d in cands))
    calls["shadow_candidate"] = sh

    # ── Family table: representative, counts ────────────────────────────────
    unit2fam, fam_rows = {}, []
    tier_rank = {"High": 0, "Partial": 1, "Low": 2}
    n_samples_by_key = calls[calls.sample_pass].groupby("canon_key")["sample"].apply(set)
    ordered = sorted(groups.values(),
                     key=lambda us: -len(set().union(*(n_samples_by_key[u] for u in us))))
    for n, us in enumerate(ordered, 1):
        fid = f"F{n:03d}"
        for u in us:
            unit2fam[u] = fid
    calls["family_id"] = calls.canon_key.map(unit2fam).fillna("")
    calls.loc[~calls.sample_pass, "family_id"] = ""

    ok = calls[calls.sample_pass]
    for fid, d in ok.groupby("family_id"):
        # Representative: canonical unit seen as High in most genomes, then in most genomes, then most general (core of the system)
        stats = d.groupby(["canon_key", "canon_pos0"]).agg(
            n_high=("tier", lambda t: (t == "High").sum()),
            n=("sample", "nunique")).reset_index()
        stats["b"] = stats.canon_key.map(bits)
        stats = stats.sort_values(["n_high", "n", "b"], ascending=[False, False, True])
        rep = stats.iloc[0]
        best_tier = d.groupby("sample").tier.apply(lambda t: min(t, key=tier_rank.get))
        shadow_share = (d.shadow_candidate != "").mean()
        fam_rows.append(dict(
            family_id=fid,
            representative=rep.canon_key,
            rep_pos0=int(rep.canon_pos0),
            modificationType=",".join(sorted(d.modificationType.unique())),
            mod_positions=",".join(map(str, sorted(d[d.canon_key == rep.canon_key].canon_pos0.unique()))),
            n_variants=d.canon_key.nunique(),
            variants=",".join(sorted(d.canon_key.unique())),
            n_genomes=d["sample"].nunique(),
            n_genomes_high=(best_tier == "High").sum(),
            n_genomes_partial=(best_tier == "Partial").sum(),
            n_genomes_low=(best_tier == "Low").sum(),
            median_fraction=d.fraction.median(),
            median_ipd=d.meanIpdRatio.median(),
            shadow_candidate_share=round(shadow_share, 3),
            shadow_examples=";".join(sorted(set(";".join(d.shadow_candidate).split(";")) - {""}))[:200],
        ))
    catalog = pd.DataFrame(fam_rows).sort_values(["n_genomes_high", "n_genomes"], ascending=False)

    # Phase 2 status (Phase 3 confirms shadows with called sites)
    def status(r):
        if r.shadow_candidate_share >= 0.5:
            return "candidate_shadow"
        if r.n_genomes_high > 0:
            return "retain"
        if r.n_genomes_partial > 0:
            return "partial_check_site_level"
        return "low_exclude"
    catalog["phase2_status"] = catalog.apply(status, axis=1)

    # ── Genome × family matrix (High tier only, pass samples) ───────────────
    hi = ok[ok.tier == "High"]
    mat = pd.crosstab(hi["sample"], hi["family_id"]).clip(upper=1)
    mat = mat.reindex(sorted(keep), fill_value=0)

    # Sanity check: two High families in one genome that are sequence-compatible at the
    # modified base would mean one MTase system counted twice
    for smp, d in hi.groupby("sample"):
        fam_units = d.drop_duplicates("family_id")[["family_id", "motif_trim", "pos0"]].values
        for i in range(len(fam_units)):
            for j in range(i + 1, len(fam_units)):
                (fa, ma, pa), (fb, mb, pb) = fam_units[i], fam_units[j]
                for bb, pbb in ((mb, pb), (revcomp(mb), len(mb) - 1 - pb)):
                    a2, b2 = align_on_modbase(ma, pa, bb, pbb)
                    if intersect(a2, b2) is not None and is_subsumed(ma, pa, mb, pb):
                        print(f"WARNING {smp}: High families {fa} ({ma}) and {fb} ({mb}) overlap")

    # ── Write outputs ───────────────────────────────────────────────────────
    cols = ["sample", "sample_pass", "motifString", "modificationType", "centerPos", "pos0_raw",
            "motif_trim", "pos0", "canon_key", "canon_pos0", "canon_orient", "is_palindrome",
            "partnerMotifString", "rc_partner_in_genome", "fraction", "nDetected", "nGenome",
            "meanScore", "meanIpdRatio", "meanCoverage", "tier", "family_id", "shadow_candidate"]
    calls[cols].to_csv(f"{args.outdir}/motif_calls_annotated.tsv", sep="\t", index=False)
    catalog.to_csv(f"{args.outdir}/motif_catalog.tsv", sep="\t", index=False)
    mat.to_csv(f"{args.outdir}/genome_family_matrix.tsv", sep="\t")

    # ── Figures ──────────────────────────────────────────────────────────────
    colors = {"High": "#2a6f97", "Partial": "#e9a03b", "Low": "#b0b0b0"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for t in ["Low", "Partial", "High"]:
        axes[0].hist(ok[ok.tier == t].fraction, bins=np.linspace(0, 1, 41), color=colors[t], label=t)
    axes[0].set(xlabel="motif methylated fraction (motifMaker)", ylabel="motif calls",
                title="Fraction by tier (pass samples)")
    axes[0].legend(frameon=False)
    for mt, mk in [("m6A", "o"), ("m4C", "s"), ("modified_base", "^")]:
        d = ok[ok.modificationType == mt]
        axes[1].scatter(d.fraction, d.meanIpdRatio, c=d.tier.map(colors), marker=mk, s=18,
                        alpha=.8, label=mt, edgecolor="none")
    axes[1].set(xlabel="fraction", ylabel="mean IPD ratio", title="Kinetic signal vs fraction")
    axes[1].legend(frameon=False, title="mod type")
    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{args.outdir}/figures/fig1_fraction_tiers.png", dpi=200)
    fig.savefig(f"{args.outdir}/figures/fig1_fraction_tiers.pdf")

    # Best tier per genome × family heatmap (families present in >= 1 genome as Partial/High)
    bt = ok.groupby(["sample", "family_id"]).tier.apply(lambda t: min(t, key=tier_rank.get)) \
        .map({"High": 2, "Partial": 1, "Low": 0.5}).unstack(fill_value=0)
    fam_show = catalog[(catalog.n_genomes_high + catalog.n_genomes_partial) > 0].family_id
    bt = bt.reindex(columns=fam_show, fill_value=0)
    labels = catalog.set_index("family_id").loc[fam_show].apply(
        lambda r: f"{r.name} {r.representative} ({r.modificationType}) [{r.phase2_status}]", axis=1)
    fig, ax = plt.subplots(figsize=(max(8, .28 * len(bt)), max(5, .22 * len(fam_show))))
    from matplotlib.colors import ListedColormap, BoundaryNorm
    cmap = ListedColormap(["#ffffff", colors["Low"], colors["Partial"], colors["High"]])
    norm = BoundaryNorm([-.1, .25, .75, 1.5, 2.5], cmap.N)
    ax.imshow(bt.T.values, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_yticks(range(len(fam_show)), labels, fontsize=6)
    ax.set_xticks(range(len(bt)), bt.index, rotation=90, fontsize=6)
    ax.set_title("Best tier per genome × motif family (white = not called; grey Low; orange Partial; blue High)",
                 fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{args.outdir}/figures/fig2_family_tier_heatmap.png", dpi=200)
    fig.savefig(f"{args.outdir}/figures/fig2_family_tier_heatmap.pdf")

    print(f"Phase 1 tiers (pass samples): {ok.tier.value_counts().to_dict()}")
    print(f"Phase 2: {ok.canon_key.nunique()} canonical motifs -> {len(catalog)} families; "
          f"status {catalog.phase2_status.value_counts().to_dict()}")
    print(f"Genome x family matrix (High only): {mat.shape}")


if __name__ == "__main__":
    main()
