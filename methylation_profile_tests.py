#!/usr/bin/env python3
"""
Methylation-profile comparisons between hosts and accessory plasmids.
Input is the retained-family rescan (motif_sites_qc.py --scan retained -> motif_qc_aggregate.py),
where every retained motif family has a calibrated methylated fraction on every contig.

Profile of a contig = calibrated fractions of the retained non-GANTC families (GANTC is ~1 on
every contig and is examined separately). For distances, fractions below --present are set to 0:
families a host does not methylate sit at background (<= 0.1), and that noise would otherwise
dominate the distances. Profile distance M = mean absolute difference. Q1 differences use raw fractions.

Q1  Plasmid vs host
    - per plasmid x family: plasmid fraction - host chromosome fraction (continuous view)
    - own-host vs other-host profile distance (rank of own host; permutation of host labels)
    - outliers (|diff| > --outlier) with plasmid copy number (plasmid / chromosome coverage)
Q2  Plasmid relatedness vs methylation (plasmid pairs, same-host pairs excluded)
    - M (host methylation-profile distance), P = 1 - wGRR (max over both directions, missing = 0),
      H = 1 - host ANI/100
    - Spearman Mantel r(M,P), r(M,H) and partial Mantel r(M,P | H); permutations shuffle the host
      methylation profiles among hosts (M is a function of the two hosts)
    - sensitivity: collapse near-clonal hosts (ANI >= --clonal-ani) to one representative
    - PERMANOVA of M by MOB-suite primary cluster (groups with >= 2 distinct hosts, one plasmid
      per host per group)
Q3  Genomes with vs without accessory plasmids
    - Fisher 2x2: GANTC-only vs >= 1 extra methylated family  x  plasmid present / absent
    - Mann-Whitney on the number of extra families per genome
    - PCoA + PERMANOVA + PERMDISP (dispersion) on host profiles

Outputs: --outdir (default methyl_call/motif_QC/profile_tests/)
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/n/projects/jp2992/Plasmidome_project_MOLNG_4509"
QC = f"{BASE}/methyl_call/motif_QC"
GANTC = "GANTC"


# ── Generic helpers ──────────────────────────────────────────────────────────
def mad_dist(X):
    """Pairwise mean absolute difference between rows of X (NaN-aware: shared features only)."""
    X = np.asarray(X, float)
    n = len(X)
    D = np.zeros((n, n))
    for i in range(n):
        d = np.abs(X[i] - X)
        D[i] = np.nanmean(np.where(np.isnan(d), np.nan, d), axis=1)
    return (D + D.T) / 2


def pcoa(D, k=2):
    """Classical multidimensional scaling."""
    n = len(D)
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J
    w, v = np.linalg.eigh(B)
    idx = np.argsort(w)[::-1]
    w, v = w[idx], v[:, idx]
    pos = w > 1e-12
    expl = w[:k] / w[pos].sum() if pos.any() else np.zeros(k)
    return v[:, :k] * np.sqrt(np.clip(w[:k], 0, None)), expl


def pseudo_f(D, g):
    """PERMANOVA pseudo-F (Anderson 2001) for distance matrix D and labels g."""
    n = len(D)
    labs = np.unique(g)
    a = len(labs)
    D2 = D ** 2
    sst = D2[np.triu_indices(n, 1)].sum() / n
    ssw = 0.0
    for l in labs:
        m = np.where(g == l)[0]
        if len(m) > 1:
            ssw += D2[np.ix_(m, m)][np.triu_indices(len(m), 1)].sum() / len(m)
    ssa = sst - ssw
    return (ssa / (a - 1)) / (ssw / (n - a)) if ssw > 0 else np.inf, ssa / sst


def permanova(D, g, perms, rng):
    f, r2 = pseudo_f(D, g)
    null = np.array([pseudo_f(D, rng.permutation(g))[0] for _ in range(perms)])
    return f, r2, (1 + np.sum(null >= f)) / (1 + perms)


def permdisp(D, g, perms, rng):
    """Dispersion test: ANOVA F on distances to group centroid in full PCoA space."""
    coords, _ = pcoa(D, k=len(D) - 1)
    def fstat(lab):
        z = np.zeros(len(D))
        for l in np.unique(lab):
            m = lab == l
            z[m] = np.linalg.norm(coords[m] - coords[m].mean(0), axis=1)
        groups = [z[lab == l] for l in np.unique(lab)]
        return stats.f_oneway(*groups).statistic, {l: z[lab == l].mean() for l in np.unique(lab)}
    f, means = fstat(g)
    null = np.array([fstat(rng.permutation(g))[0] for _ in range(perms)])
    return f, means, (1 + np.sum(null >= f)) / (1 + perms)


def resid(y, x):
    """Residuals of y after linear regression on x (with intercept)."""
    A = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return y - A @ beta


def mantel_suite(pairs, host_prof, P, H, perms, rng):
    """Spearman Mantel r(M,P), r(M,H), partial r(M,P|H). M is rebuilt from (permuted) host profiles."""
    hi, hj = pairs[:, 0], pairs[:, 1]
    hosts = list(host_prof.index)
    def M_of(prof):
        X = prof.values
        return np.nanmean(np.abs(X[hi] - X[hj]), axis=1)
    rP, rH = stats.rankdata(P), stats.rankdata(H)
    rPres = resid(rP, rH)
    def stat(prof):
        rM = stats.rankdata(M_of(prof))
        return (np.corrcoef(rM, rP)[0, 1], np.corrcoef(rM, rH)[0, 1],
                np.corrcoef(resid(rM, rH), rPres)[0, 1])
    obs = np.array(stat(host_prof))
    null = np.empty((perms, 3))
    for b in range(perms):
        perm = host_prof.iloc[rng.permutation(len(hosts))]
        perm.index = host_prof.index
        null[b] = stat(perm)
    p = [(1 + np.sum(null[:, k] >= obs[k])) / (1 + perms) for k in range(3)]
    return obs, p


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--long", default=f"{QC}/phase5_retained/contig_family_long.tsv")
    ap.add_argument("--samples", default=f"{QC}/samples_pass.txt")
    ap.add_argument("--tags", default=f"{BASE}/contigs_to_tags.tsv")
    ap.add_argument("--flye-dir", default=f"{BASE}/flye_asm")
    ap.add_argument("--host-ani", default=f"{BASE}/strain_all_vs_all_ANI/strain_all_vs_all_ANI.tsv")
    ap.add_argument("--wgrr", default=f"{BASE}/plasmids_decontamination/mobmess_all_22072026/wgrr_network/wGGR.tsv")
    ap.add_argument("--skani", default=f"{BASE}/plasmids_decontamination/mobmess_all_22072026/all_vs_all_ANI_acc_pls/acc_pls_matrix_skani.tsv")
    ap.add_argument("--clusters", default=f"{BASE}/plasmids_decontamination/mobmess_all_22072026/repabc_analysis/all_seqs_w_repl_prot_genomad.csv")
    ap.add_argument("--outdir", default=f"{QC}/profile_tests")
    ap.add_argument("--min-sites", type=int, default=10)
    ap.add_argument("--present", type=float, default=0.5)
    ap.add_argument("--outlier", type=float, default=0.10)
    ap.add_argument("--clonal-ani", type=float, default=99.99)
    ap.add_argument("--perms", type=int, default=9999)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    os.makedirs(f"{args.outdir}/figures", exist_ok=True)
    rng = np.random.default_rng(args.seed)
    out = []                                                   # summary rows

    def note(test, stat, value, p=np.nan, n=np.nan, detail=""):
        out.append(dict(test=test, statistic=stat, value=value, p_value=p, n=n, detail=detail))
        print(f"{test:<42} {stat:<26} {value:>9.4g}  p={p:.4g}  n={n}  {detail}")

    # ── Load profiles ────────────────────────────────────────────────────────
    samples = [l.strip() for l in open(args.samples) if l.strip()]
    L = pd.read_csv(args.long, sep="\t")
    L = L[(L.final_status == "retain") & (L.n_covered >= args.min_sites)]
    fams = L.drop_duplicates("family_id").set_index("family_id").representative
    gantc_fid = fams[fams == GANTC].index[0]
    feat = [f for f in sorted(fams.index) if f != gantc_fid]
    F = L.pivot_table(index="contig", columns="family_id", values="fraction")
    F = F.reindex(columns=sorted(fams.index))
    tags = pd.read_csv(args.tags, sep="\t", header=None, names=["contig", "replicon"])
    meta = tags.set_index("contig").replicon
    contig_sample = L.drop_duplicates("contig").set_index("contig")["sample"]
    chrom = {s: c for c, s in contig_sample.items() if meta.get(c) == "Chromosome"}
    Fd = F.where(F >= args.present, 0.0).where(F.notna())                      # denoised, for distances
    host_prof = pd.DataFrame({s: Fd.loc[chrom[s], feat] for s in samples}).T    # hosts x families
    acc = [c for c in F.index if meta.get(c) == "Other" and contig_sample[c] in samples]
    host_of = {c: contig_sample[c] for c in acc}
    print(f"{len(samples)} hosts, {len(acc)} accessory plasmids, {len(feat)} non-GANTC retained families")

    # Copy number proxy: contig coverage / chromosome coverage (Flye assembly_info)
    cov = {}
    for s in samples:
        ai = pd.read_csv(f"{args.flye_dir}/{s}_filtered/assembly_info.txt", sep="\t")
        for r in ai.itertuples():
            cov[f"{r[1]}_{s}_filtered"] = r[3]
    copy_num = {c: cov.get(c, np.nan) / cov.get(chrom[host_of[c]], np.nan) for c in acc}

    # ── Q1: plasmid vs host ─────────────────────────────────────────────────
    rows = []
    for c in acc:
        h = host_of[c]
        for f in fams.index:
            pf, hf = F.at[c, f], F.at[chrom[h], f]
            rows.append(dict(plasmid=c, host=h, family_id=f, motif=fams[f], plasmid_frac=pf, host_frac=hf,
                             host_methylated=hf >= args.present, diff=pf - hf,
                             length=L.loc[L.contig == c, "length"].iloc[0], copy_number=copy_num[c]))
    q1 = pd.DataFrame(rows)
    q1.to_csv(f"{args.outdir}/q1_plasmid_vs_host_by_family.tsv", sep="\t", index=False)
    meth = q1[q1.host_methylated & q1["diff"].notna()]
    unmeth = q1[~q1.host_methylated & q1.plasmid_frac.notna()]
    note("Q1 families methylated in host", "median plasmid-host diff", meth["diff"].median(), n=len(meth),
         detail=f"range {meth['diff'].min():.3f}..{meth['diff'].max():.3f}")
    note("Q1 families not methylated in host", "max plasmid fraction", unmeth.plasmid_frac.max(), n=len(unmeth),
         detail=f"plasmids >= {args.present}: {(unmeth.plasmid_frac >= args.present).sum()}")
    outl = meth[meth["diff"].abs() > args.outlier].sort_values("diff")
    outl.to_csv(f"{args.outdir}/q1_outliers.tsv", sep="\t", index=False)

    # Own-host vs other-host distance
    P_prof = Fd.loc[acc, feat]
    own, ranks = [], []
    for c in acc:
        d = np.nanmean(np.abs(host_prof.values - P_prof.loc[c].values), axis=1)
        d = pd.Series(d, index=host_prof.index)
        own.append(d[host_of[c]])
        ranks.append((stats.rankdata(d)[list(d.index).index(host_of[c])] - 1) / (len(d) - 1))
    own = np.array(own)
    note("Q1 own-host profile distance", "median", np.median(own), n=len(acc),
         detail=f"own-host rank percentile among 44 hosts (0 = closest): median {np.median(ranks):.3f}, "
                f"max {np.max(ranks):.3f}")
    hosts_arr = np.array([host_of[c] for c in acc])
    def mean_own(hlabels):
        return np.mean([np.nanmean(np.abs(host_prof.loc[h].values - P_prof.loc[c].values)) for c, h in zip(acc, hlabels)])
    obs = mean_own(hosts_arr)
    null = np.array([mean_own(rng.permutation(hosts_arr)) for _ in range(min(args.perms, 1999))])
    note("Q1 own-host vs shuffled host", "mean own distance", obs, (1 + np.sum(null <= obs)) / (1 + len(null)),
         len(acc), detail=f"null mean {null.mean():.4f}; ties from GANTC-only hosts make this mostly a host-diversity test")

    # ── Q2: Mantel / partial Mantel over plasmid pairs ─────────────────────
    w = pd.read_csv(args.wgrr, sep="\t", usecols=["query", "subject", "wGRR"])
    accset = set(acc)
    w = w[w["query"].isin(accset) & w["subject"].isin(accset)]
    wmax = {}
    for q, s_, v in zip(w["query"], w["subject"], w.wGRR):
        k = tuple(sorted((q, s_)))
        wmax[k] = max(v, wmax.get(k, 0.0))
    in_w = sorted(set(w["query"]) | set(w["subject"]))
    plas = [c for c in acc if c in set(in_w)]
    missing = sorted(set(acc) - set(plas))
    print(f"Q2: {len(plas)} plasmids with wGRR (not in wGRR table: {missing})")

    ani = pd.read_csv(args.host_ani, sep="\t", header=None, usecols=[0, 1, 2], names=["a", "b", "ani"])
    getname = lambda p: os.path.basename(p).replace("_filtered_asm.fasta", "").replace("_asm.fasta", "")
    ani["a"], ani["b"] = ani.a.map(getname), ani.b.map(getname)
    ani = ani[ani.a.isin(samples) & ani.b.isin(samples)]
    A = ani.groupby(["a", "b"]).ani.mean().unstack().reindex(index=samples, columns=samples)
    Av = (A.values + A.values.T) / 2
    np.fill_diagonal(Av, 100.0)
    A = pd.DataFrame(Av, index=samples, columns=samples)
    hosts_q2 = list(host_prof.index)
    hidx = {h: i for i, h in enumerate(hosts_q2)}

    def run_q2(plist, label):
        pr = [(i, j) for i in range(len(plist)) for j in range(i + 1, len(plist))
              if host_of[plist[i]] != host_of[plist[j]]]                           # drop same-host pairs
        pairs = np.array([[hidx[host_of[plist[i]]], hidx[host_of[plist[j]]]] for i, j in pr])
        Pd = np.array([1 - wmax.get(tuple(sorted((plist[i], plist[j]))), 0.0) for i, j in pr])
        Hd = np.array([1 - A.at[host_of[plist[i]], host_of[plist[j]]] / 100 for i, j in pr])
        if np.isnan(Hd).any():
            print(f"  warning: {np.isnan(Hd).sum()} host pairs without ANI; set to max observed distance")
            Hd = np.where(np.isnan(Hd), np.nanmax(Hd), Hd)
        obs, p = mantel_suite(pairs, host_prof, Pd, Hd, args.perms, rng)
        n = f"{len(plist)} plasmids / {len(pr)} pairs"
        note(f"Q2 {label}", "Mantel r(M,P) Spearman", obs[0], p[0], n)
        note(f"Q2 {label}", "Mantel r(M,H) Spearman", obs[1], p[1], n)
        note(f"Q2 {label}", "partial r(M,P|H)", obs[2], p[2], n)
        return pr, Pd, Hd, pairs

    pr, Pd, Hd, pairs = run_q2(plas, "all hosts")
    pd.DataFrame({"plasmid_i": [plas[i] for i, _ in pr], "plasmid_j": [plas[j] for _, j in pr],
                  "host_i": [host_of[plas[i]] for i, _ in pr], "host_j": [host_of[plas[j]] for _, j in pr],
                  "M": np.nanmean(np.abs(host_prof.values[pairs[:, 0]] - host_prof.values[pairs[:, 1]]), axis=1),
                  "P_1_minus_wGRR": Pd, "H_1_minus_ANI": Hd}).to_csv(f"{args.outdir}/q2_pairs.tsv", sep="\t", index=False)

    # Near-clonal hosts collapsed to one representative (keeps that host's plasmids only)
    rep, done = {}, set()
    for h in sorted(hosts_q2):
        if h in done:
            continue
        grp = [x for x in hosts_q2 if A.at[h, x] >= args.clonal_ani and x not in done]
        for x in grp:
            rep[x] = h
            done.add(x)
    clonal_groups = {r: [x for x in rep if rep[x] == r] for r in set(rep.values())}
    clonal_groups = {r: g for r, g in clonal_groups.items() if len(g) > 1}
    print(f"Near-clonal host groups (ANI >= {args.clonal_ani}): {clonal_groups}")
    plas_c = [c for c in plas if rep[host_of[c]] == host_of[c]]
    run_q2(plas_c, f"clonal hosts collapsed (ANI>={args.clonal_ani})")

    # PERMANOVA by MOB-suite primary cluster
    cl = pd.read_csv(args.clusters, usecols=["contig", "primary_cluster_id"]).drop_duplicates("contig").set_index("contig").primary_cluster_id
    pc = pd.DataFrame({"plasmid": plas, "host": [host_of[c] for c in plas], "group": [cl.get(c, np.nan) for c in plas]}).dropna()
    pc = pc.drop_duplicates(["group", "host"])                                    # one plasmid per host per group
    ok = pc.groupby("group").host.nunique()
    pc = pc[pc.group.isin(ok[ok >= 2].index)]
    if pc.group.nunique() >= 2:
        Dm = mad_dist(host_prof.loc[pc.host].values)
        f, r2, p = permanova(Dm, pc.group.values, args.perms, rng)
        note("Q2 PERMANOVA M ~ MOB-suite primary cluster", "pseudo-F", f, p, len(pc),
             detail=f"R2={r2:.3f}; groups: {pc.group.value_counts().to_dict()}")

    # ── Q3: genomes with vs without accessory plasmids ─────────────────────
    has_pl = pd.Series({s: any(meta.get(c) == "Other" for c in tags.contig if c.endswith(f"_{s}_filtered")) for s in samples})
    n_extra = (host_prof >= args.present).sum(axis=1)
    g3 = pd.DataFrame({"has_accessory_plasmid": has_pl, "n_extra_families": n_extra})
    g3.to_csv(f"{args.outdir}/q3_genomes.tsv", sep="\t")
    tab = pd.crosstab(g3.has_accessory_plasmid, g3.n_extra_families > 0)
    orr, pf = stats.fisher_exact(tab.values)
    note("Q3 Fisher: plasmid x extra RM system", "odds ratio", orr, pf, len(g3),
         detail=f"table [no plasmid / plasmid] x [GANTC-only / >=1 extra]: {tab.values.tolist()}")
    u = stats.mannwhitneyu(n_extra[has_pl], n_extra[~has_pl], alternative="two-sided")
    note("Q3 Mann-Whitney n extra families", "U", u.statistic, u.pvalue, len(g3),
         detail=f"median with={n_extra[has_pl].median()}, without={n_extra[~has_pl].median()}")
    Dh = mad_dist(host_prof.values)
    lab = np.where(has_pl.values, "with", "without")
    f, r2, p = permanova(Dh, lab, args.perms, rng)
    note("Q3 PERMANOVA host profile ~ plasmid status", "pseudo-F", f, p, len(lab), detail=f"R2={r2:.3f}")
    fd, means, pd_ = permdisp(Dh, lab, min(args.perms, 1999), rng)
    note("Q3 PERMDISP (dispersion)", "F", fd, pd_, len(lab), detail=f"mean dist to centroid {means}")

    pd.DataFrame(out).to_csv(f"{args.outdir}/results_summary.tsv", sep="\t", index=False)

    # ── Figures ──────────────────────────────────────────────────────────────
    blue, orange, grey = "#2a78d6", "#e9a03b", "#9e9e9e"
    # Fig 1: plasmid - host difference for host-methylated families
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.4, 1]})
    order = meth.groupby("motif").size().sort_values(ascending=False).index
    for i, mo in enumerate(order):
        d = meth[meth.motif == mo]
        ax[0].scatter(np.full(len(d), i) + rng.uniform(-.18, .18, len(d)), d["diff"], s=16,
                      color=blue, edgecolor="white", linewidth=.5, zorder=3)
    for r in outl.itertuples():
        ax[0].annotate(f"{r.plasmid.replace('_filtered', '')}\n(copy {r.copy_number:.1f}x)",
                       (list(order).index(r.motif), r.diff), xytext=(6, 0), textcoords="offset points",
                       fontsize=6, color="#555555", va="center")
    ax[0].axhline(0, color=grey, lw=.8)
    ax[0].set_xticks(range(len(order)), order, rotation=60, ha="right", fontsize=7, family="monospace")
    ax[0].set(ylabel="plasmid − host chromosome fraction", title="Families methylated in the host")
    ax[1].hist(unmeth.plasmid_frac, bins=np.linspace(0, 1, 41), color=grey)
    ax[1].axvline(args.present, ls="--", color=orange)
    ax[1].set(xlabel="plasmid methylated fraction", ylabel="plasmid × family", title="Families NOT methylated in the host")
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{args.outdir}/figures/q1_plasmid_vs_host.png", dpi=250)
    fig.savefig(f"{args.outdir}/figures/q1_plasmid_vs_host.pdf")

    # Fig 2: Mantel scatter
    Mv = np.nanmean(np.abs(host_prof.values[pairs[:, 0]] - host_prof.values[pairs[:, 1]]), axis=1)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].scatter(Pd, Mv + rng.uniform(-.004, .004, len(Mv)), s=8, alpha=.5, color=blue, edgecolor="none")
    ax[0].set(xlabel="plasmid distance (1 − wGRR)", ylabel="host methylation-profile distance (M)")
    ax[1].scatter(Hd, Mv + rng.uniform(-.004, .004, len(Mv)), s=8, alpha=.5, color=blue, edgecolor="none")
    ax[1].set(xlabel="host distance (1 − ANI/100)", ylabel="M")
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Plasmid pairs in different hosts (points jittered vertically)", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{args.outdir}/figures/q2_mantel_scatter.png", dpi=250)
    fig.savefig(f"{args.outdir}/figures/q2_mantel_scatter.pdf")

    # Fig 3: PCoA of host profiles (identical profiles stacked; size = number of genomes)
    xy, expl = pcoa(Dh)
    pts = pd.DataFrame({"x": xy[:, 0].round(6), "y": xy[:, 1].round(6), "g": lab})
    agg = pts.groupby(["x", "y", "g"]).size().reset_index(name="n")
    fig, ax = plt.subplots(figsize=(6, 5))
    for g, col, dx in (("with", blue, -0.004), ("without", orange, 0.004)):
        d = agg[agg.g == g]
        ax.scatter(d.x + dx, d.y, s=30 + 25 * d.n, color=col, alpha=.8, edgecolor="white", label=f"{g} accessory plasmids")
        for r in d[d.n > 1].itertuples():
            ax.annotate(str(r.n), (r.x + dx, r.y), ha="center", va="center", fontsize=7, color="white")
    ax.set(xlabel=f"PCoA 1 ({expl[0]:.0%})", ylabel=f"PCoA 2 ({expl[1]:.0%})",
           title="Host methylation profiles (non-GANTC retained families)")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{args.outdir}/figures/q3_host_pcoa.png", dpi=250)
    fig.savefig(f"{args.outdir}/figures/q3_host_pcoa.pdf")
    print("done")


if __name__ == "__main__":
    main()
