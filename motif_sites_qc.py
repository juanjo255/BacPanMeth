#!/usr/bin/env python3
"""
Methylation motif QC — Phases 3–4 worker (one genome per call; run as a SLURM array).
See methylation_motif_QC_methodology.md.

For every motif called by motifMaker in this sample (rows of motif_calls_annotated.tsv):
  Phase 3  enumerate all motif sites on both strands of the calling assembly, join them to
           detected modified bases (basemods.gff), run the m6A-shadow test and the
           within-family site-overlap check
  Phase 4  attach per-base kinetics (basemods.csv) to every site, including undetected
           ones, and summarise per motif x replicon

Coordinates: GFF and basemods.csv `tpl` are 1-based; basemods.csv strand 0 = '+', 1 = '-'
(checked: tpl 1 / strand 1 / score 39 == GFF pos 1 '-' score 39 in Liana_087_2).
Sites spanning the origin of circular contigs are not enumerated (<= motif length per contig).

Outputs in --outdir:
  <s>.sites.tsv.gz                  one row per motif site (all tiers)
  <s>.motif_replicon_summary.tsv    per motif x replicon detection / kinetics summary
  <s>.shadow_test.tsv               per non-High-m6A motif: offset to nearest High m6A site
  <s>.family_overlap.tsv            per family variant: share of called sites shared with the family reference
  <s>.m6A_footprint.tsv             kinetics at -5..+5 (both strands) around High m6A sites, by base:
                                    the independent test of m6A "shadow" calls (e.g. IPD at the N of
                                    GANTC when N = C vs A/G/T)

Note: `detected` = any basemods.gff row at the site (any type, any score). It is looser than
motifMaker's per-motif fraction and must be calibrated (score cutoff, background rate) in Phase 5.
"""

import argparse
import math
import os
import re
import time
from collections import Counter

import numpy as np
import pandas as pd
from Bio import SeqIO

BASE = "/n/projects/jp2992/Plasmidome_project_MOLNG_4509"

# ── Tunable thresholds (mirrored in the methodology doc) ─────────────────────
MIN_SITE_COV     = 25     # Phase 4: site counts as "covered"
UNMETH_IPD       = 1.5    # Phase 4: covered site with IPD ratio below this = unmethylated
SHADOW_MAX_DIST  = 5      # Phase 3: window around High m6A sites
SHADOW_MIN_SHARE = 0.80   # Phase 3: share of called sites at one offset to call a shadow
CSV_CHUNK        = 5_000_000

IUPAC_RE = {"A": "A", "C": "C", "G": "G", "T": "T", "R": "[AG]", "Y": "[CT]", "S": "[CG]",
            "W": "[AT]", "K": "[GT]", "M": "[AC]", "B": "[CGT]", "D": "[AGT]", "H": "[ACT]",
            "V": "[ACG]", "N": "[ACGT]"}
COMP = {"A": "T", "C": "G", "G": "C", "T": "A", "R": "Y", "Y": "R", "S": "S", "W": "W",
        "K": "M", "M": "K", "B": "V", "V": "B", "D": "H", "H": "D", "N": "N"}


def revcomp(m):
    return "".join(COMP[c] for c in reversed(m))


def motif_regex(m):
    # Lookahead so overlapping occurrences are all reported
    return re.compile("(?=" + "".join(IUPAC_RE[c] for c in m) + ")")


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def log(msg, t0=[time.time()]):
    print(f"[{time.time() - t0[0]:7.1f}s] {msg}", flush=True)


def scan_sites(contigs, motif, pos0, label):
    """All modified-base positions (1-based) of `motif` on both strands."""
    L = len(motif)
    fw, rv = motif_regex(motif), motif_regex(revcomp(motif))
    out = []
    for name, seq in contigs.items():
        p_fw = np.fromiter((m.start() for m in fw.finditer(seq)), dtype=np.int64)
        p_rv = np.fromiter((m.start() for m in rv.finditer(seq)), dtype=np.int64)
        out.append(pd.DataFrame({"contig": name, "pos": p_fw + pos0 + 1, "strand": "+"}))
        out.append(pd.DataFrame({"contig": name, "pos": p_rv + (L - 1 - pos0) + 1, "strand": "-"}))
    df = pd.concat(out, ignore_index=True)
    df["motif_id"] = label
    return df


def footprint_keys(m6a):
    """Positions at offsets -SHADOW_MAX_DIST..+SHADOW_MAX_DIST (m6A-strand orientation) on the same
    and opposite strand of every High m6A site."""
    sgn = np.where(m6a.strand.values == "+", 1, -1)
    flip = np.where(m6a.strand.values == "+", "-", "+")
    out = []
    for off in range(-SHADOW_MAX_DIST, SHADOW_MAX_DIST + 1):
        for rel in ("same", "opposite"):
            out.append(pd.DataFrame({
                "motif_id": m6a.motif_id.values, "contig": m6a.contig.values,
                "pos": m6a.pos.values + sgn * off,
                "strand": m6a.strand.values if rel == "same" else flip,
                "offset": off, "rel_strand": rel}))
    return pd.concat(out, ignore_index=True)


def base_at(contigs, df):
    """Base on the given strand at 1-based positions (N if outside the contig)."""
    comp = str.maketrans("ACGT", "TGCA")
    bases = np.full(len(df), "N", dtype=object)
    for c, idx in df.groupby("contig").groups.items():
        seq = contigs[c]
        pos = df.loc[idx, "pos"].values
        ok = (pos >= 1) & (pos <= len(seq))
        b = np.array(["N"] * len(pos), dtype=object)
        b[ok] = [seq[p - 1] for p in pos[ok]]
        minus = df.loc[idx, "strand"].values == "-"
        b[minus] = [x.translate(comp) for x in b[minus]]
        bases[df.index.get_indexer(idx)] = b
    return bases


def read_gff(path):
    """Detected modified bases -> DataFrame keyed by contig/pos/strand."""
    rows = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9:
                continue
            attrs = dict(kv.split("=", 1) for kv in f[8].split(";") if "=" in kv)
            rows.append((f[0], int(f[3]), f[6], f[2], int(f[5]),
                         float(attrs.get("IPDRatio", "nan")), float(attrs.get("frac", "nan"))))
    return pd.DataFrame(rows, columns=["contig", "pos", "strand", "call_type", "call_score",
                                       "call_ipd", "call_frac"])


def read_basemods_csv(path, keys):
    """Stream the per-base CSV and keep only rows at motif sites."""
    need = keys.set_index(["contig", "pos", "strand"]).index
    parts = []
    reader = pd.read_csv(path, usecols=["refName", "tpl", "strand", "score", "ipdRatio", "coverage"],
                         dtype={"refName": "category", "tpl": np.int64, "strand": np.int8,
                                "score": np.float32, "ipdRatio": np.float32, "coverage": np.float32},
                         chunksize=CSV_CHUNK)
    for i, ch in enumerate(reader):
        ch = ch.rename(columns={"refName": "contig", "tpl": "pos"})
        ch["contig"] = ch["contig"].astype(str)
        ch["strand"] = np.where(ch["strand"] == 0, "+", "-")
        idx = pd.MultiIndex.from_frame(ch[["contig", "pos", "strand"]])
        parts.append(ch[idx.isin(need)])
        log(f"  basemods.csv chunk {i + 1}: kept {len(parts[-1]):,}")
    return pd.concat(parts, ignore_index=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", required=True)
    ap.add_argument("--qc-dir", default=f"{BASE}/methyl_call/motif_QC")
    ap.add_argument("--methyl-dir", default=f"{BASE}/methyl_call/ipdsummary_out")
    ap.add_argument("--flye-dir", default=f"{BASE}/flye_asm")
    ap.add_argument("--tags", default=f"{BASE}/contigs_to_tags.tsv")
    ap.add_argument("--outdir", default=f"{BASE}/methyl_call/motif_QC/motif_sites")
    ap.add_argument("--contigs", nargs="*", help="restrict to these contigs (smoke tests)")
    ap.add_argument("--scan", choices=["called", "retained"], default="called",
                    help="called: motifs motifMaker called in this genome; retained: every strand-specific "
                         "High-tier unit of the retained families (High in >= 1 genome), in every genome")
    args = ap.parse_args()
    s = args.sample
    os.makedirs(args.outdir, exist_ok=True)

    # ── Inputs ───────────────────────────────────────────────────────────────
    allc = pd.read_csv(f"{args.qc_dir}/motif_calls_annotated.tsv", sep="\t")
    allc = allc[allc["sample_pass"]].copy()
    allc["motif_id"] = allc["motifString"] + "|" + allc["modificationType"]
    calls = allc[allc["sample"] == s].reset_index(drop=True)          # this genome's own calls
    if calls.empty:
        raise SystemExit(f"{s}: no passing motif calls in motif_calls_annotated.tsv")
    if args.scan == "called":
        scan = calls
    else:
        # Every High-tier unit seen in any genome; tier = this genome's own tier, or not_called
        scan = allc[allc.tier == "High"].drop_duplicates("motif_id")[
            ["motif_id", "motifString", "motif_trim", "pos0", "modificationType", "family_id"]].copy()
        own = calls.set_index("motif_id").tier
        scan["tier"] = scan.motif_id.map(own).fillna("not_called")
        scan = scan.reset_index(drop=True)

    fasta = f"{args.flye_dir}/{s}_filtered/{s}_filtered_asm.fasta"
    contigs = {r.id: str(r.seq).upper() for r in SeqIO.parse(fasta, "fasta")}
    if args.contigs:
        contigs = {k: v for k, v in contigs.items() if k in set(args.contigs)}
    tags = pd.read_csv(args.tags, sep="\t", header=None, names=["contig", "replicon"])
    replicon = dict(zip(tags.contig, tags.replicon))
    log(f"{s}: scanning {len(scan)} motifs ({args.scan}), {len(contigs)} contigs ({sum(map(len, contigs.values())):,} bp)")

    # ── Phase 3.1: enumerate sites ───────────────────────────────────────────
    sites = pd.concat([scan_sites(contigs, r.motif_trim, int(r.pos0), r.motif_id)
                       for r in scan.itertuples()], ignore_index=True)
    log(f"sites enumerated: {len(sites):,}")

    # ── Phase 3.2: join detected calls ───────────────────────────────────────
    gff = read_gff(f"{args.methyl_dir}/{s}/{s}_basemods.gff")
    if args.contigs:
        gff = gff[gff.contig.isin(contigs)]
    log(f"GFF calls: {len(gff):,}")
    sites = sites.merge(gff, on=["contig", "pos", "strand"], how="left")
    sites["detected"] = sites["call_type"].notna()

    # High m6A motif sites + their -5..+5 footprint (both strands)
    hi_ids = calls[(calls.tier == "High") & (calls.modificationType == "m6A")].motif_id
    m6a = sites[sites.motif_id.isin(hi_ids)]
    fp = footprint_keys(m6a)
    fp = fp[fp.pos >= 1].reset_index(drop=True)
    log(f"footprint positions around {len(m6a):,} High m6A sites: {len(fp):,}")

    # ── Phase 4: per-base kinetics at all sites (+ footprint positions) ─────
    uniq = pd.concat([sites[["contig", "pos", "strand"]], fp[["contig", "pos", "strand"]]]).drop_duplicates()
    kin = read_basemods_csv(f"{args.methyl_dir}/{s}/{s}_basemods.csv", uniq)
    sites = sites.merge(kin, on=["contig", "pos", "strand"], how="left")
    sites["replicon"] = sites["contig"].map(replicon).fillna("Unassigned")
    sites["covered"] = sites["coverage"] >= MIN_SITE_COV
    sites["unmethylated"] = sites["covered"] & (sites["ipdRatio"] < UNMETH_IPD)
    meta = scan.set_index("motif_id")[["family_id", "tier", "modificationType"]]
    sites = sites.join(meta, on="motif_id")
    sites.to_csv(f"{args.outdir}/{s}.sites.tsv.gz", sep="\t", index=False, compression="gzip")
    log(f"wrote sites table ({len(sites):,} rows)")

    # ── Phase 4 summary: motif x replicon ────────────────────────────────────
    rows = []
    for (mid, rep), d in sites.groupby(["motif_id", "replicon"]):
        cov = d[d.covered]
        k, n = int(cov.detected.sum()), len(cov)
        lo, hi = wilson(k, n)
        rows.append(dict(sample=s, motif_id=mid, family_id=meta.loc[mid, "family_id"],
                         tier=meta.loc[mid, "tier"], replicon=rep, n_sites=len(d), n_covered=n,
                         n_detected=k, frac_detected=k / n if n else np.nan, ci_low=lo, ci_high=hi,
                         median_ipd_covered=cov.ipdRatio.median(),
                         n_unmethylated=int(d.unmethylated.sum()),
                         median_coverage=d.coverage.median()))
    pd.DataFrame(rows).to_csv(f"{args.outdir}/{s}.motif_replicon_summary.tsv", sep="\t", index=False)

    # ── Phase 3.3a: kinetic footprint around High m6A sites ─────────────────
    # Independent of motif calling: compares IPD at neighbouring C (candidate m4C shadow) with
    # A/G/T at the same offset. Positions that are themselves High m6A sites are excluded
    # (except offset 0 / same strand = the site itself).
    m6a = sites[sites.motif_id.isin(hi_ids)]
    m6a_keys = set(zip(m6a.contig, m6a.pos, m6a.strand))
    fp["base"] = base_at(contigs, fp)
    fp = fp.merge(kin, on=["contig", "pos", "strand"], how="left") \
           .merge(gff[["contig", "pos", "strand", "call_type"]], on=["contig", "pos", "strand"], how="left")
    fp["is_m6A_site"] = [k in m6a_keys for k in zip(fp.contig, fp.pos, fp.strand)]
    fp = fp[~fp.is_m6A_site | ((fp.offset == 0) & (fp.rel_strand == "same"))]
    fp["covered"] = fp.coverage >= MIN_SITE_COV
    fpc = fp[fp.covered]
    fps = fpc.groupby(["motif_id", "offset", "rel_strand", "base"]).agg(
        n=("ipdRatio", "size"),
        median_ipd=("ipdRatio", "median"),
        mean_ipd=("ipdRatio", "mean"),
        share_ipd_ge2=("ipdRatio", lambda x: (x >= 2).mean()),
        share_called=("call_type", lambda x: x.notna().mean()),
        share_called_m4C=("call_type", lambda x: (x == "m4C").mean())).reset_index()
    fps.insert(0, "sample", s)
    fps.to_csv(f"{args.outdir}/{s}.m6A_footprint.tsv", sep="\t", index=False)
    log(f"wrote m6A footprint ({len(fps)} rows)")

    # ── Phase 3.3b: shadow test on called sites (informative only for motifs that are
    #    NOT sequence-contained in a High m6A motif; contained ones score 1.0 by construction)
    m6a_keys = set(zip(m6a.contig, m6a.pos, m6a.strand))
    flip = {"+": "-", "-": "+"}
    rows = []
    for mid, d in sites[~sites.motif_id.isin(hi_ids)].groupby("motif_id"):
        det = d[d.detected]
        offs = Counter()
        n_near = 0
        for c, p, st in zip(det.contig, det.pos, det.strand):
            sgn = 1 if st == "+" else -1
            hit = []
            for off in range(-SHADOW_MAX_DIST, SHADOW_MAX_DIST + 1):
                for rel, st2 in (("same", st), ("opposite", flip[st])):
                    if (c, p - sgn * off, st2) in m6a_keys:   # off = m4C pos - m6A pos, in m4C-strand orientation
                        hit.append((off, rel))
            if hit:
                n_near += 1
                offs.update(hit)
        modal, modal_n = (offs.most_common(1)[0] if offs else ((None, None), 0))
        share = modal_n / len(det) if len(det) else np.nan
        rows.append(dict(sample=s, motif_id=mid, family_id=meta.loc[mid, "family_id"],
                         tier=meta.loc[mid, "tier"], n_sites=len(d), n_detected=len(det),
                         share_detected_near_m6A=n_near / len(det) if len(det) else np.nan,
                         modal_offset=modal[0], modal_strand=modal[1], share_at_modal=share,
                         shadow=bool(share >= SHADOW_MIN_SHARE) if len(det) else False,
                         high_m6A_motifs=",".join(hi_ids)))
    pd.DataFrame(rows).to_csv(f"{args.outdir}/{s}.shadow_test.tsv", sep="\t", index=False)

    # ── Phase 3.4: within-family overlap of called sites ─────────────────────
    rows = []
    for fid, d in sites[sites.detected].groupby("family_id"):
        sets = {m: set(zip(x.contig, x.pos, x.strand)) for m, x in d.groupby("motif_id")}
        ref = max(sets, key=lambda m: len(sets[m]))
        for m, st in sets.items():
            rows.append(dict(sample=s, family_id=fid, reference=ref, motif_id=m,
                             n_detected=len(st),
                             share_in_reference=len(st & sets[ref]) / len(st) if st else np.nan))
    pd.DataFrame(rows).to_csv(f"{args.outdir}/{s}.family_overlap.tsv", sep="\t", index=False)
    log("done")


if __name__ == "__main__":
    main()
