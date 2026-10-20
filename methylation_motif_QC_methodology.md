# Methylation motif QC & harmonization — methodology

Project: Plasmidome (MOLNG-4509), *S. meliloti* open pangenome
Author: J. Picón Cossio (analysis design drafted with Claude Code)
Status (2026-10-19): **Phases 0–5 run.**
- Phases 0–2: `motif_qc_harmonize.py`, run locally.
- Phases 3–4: `motif_sites_qc.py` + `motif_sites_qc_slurm.sh`, SLURM job 5445169, 44/44 tasks OK, ≤ 94 s and ≤ 3.4 GB per task.
- Phase 5: `motif_qc_aggregate.py`, run locally, 3 min, 1.4 GB.

Outputs are in `methyl_call/motif_QC/` (Phase 5 in `phase5/`, retained rescan in `phase5_retained/`, comparisons in `profile_tests/`). Thresholds marked *(tunable)* are starting values.

---

## 1. Motivation

The current genome/contig motif matrix (`methyl_call/methylation_correlation/contig_motif_matrix.csv`, built by `methylation_correlation_analysis.py`, step 2–3) marks a motif as *present* on a contig if **≥ 1 detected base** in `*_basemods_w_motif.gff` carries that motif label. There is:

- no filter on motif-level methylated fraction, site count or score;
- no merging of reverse-complement partners or degenerate variants of the same motif;
- no removal of kinetic "shadow" calls.

As a result the matrix has ~309 motif columns for 44 genomes, while the pooled `*_motif.csv` files show only a handful of motifs that are actually fully methylated. Every downstream result (HDBSCAN vs mobmess, t-SNE, dendrograms, tanglegrams) inherits this noise. This document defines a QC + harmonization step that produces a clean motif catalog to replace that matrix.

## 2. Inputs

| Input | Path | Size |
|---|---|---|
| Motif tables (motifMaker) | `methyl_call/ipdsummary_out/<s>/<s>_motif.csv` | 45 files, 364 motif rows |
| Detected modified bases | `methyl_call/ipdsummary_out/<s>/<s>_basemods.gff` (+ `_w_motif.gff`) | ~1.4 GB total |
| Per-base kinetics (all positions) | `methyl_call/ipdsummary_out/<s>/<s>_basemods.csv` | ~14 GB total (~1 GB/genome) |
| Assemblies (reference for calling) | `flye_asm/<s>_filtered/<s>_filtered_asm.fasta` | 44 genomes |
| Sample list | `final_dataset.txt` | |
| Replicon labels | `contigs_to_tags.tsv` | |
| RM/MTase annotation (later step) | `flye_asm/type_2_RM/`, `defensefinder_all_genomes/` | |

Upstream calling (for reference): `ccs-kinetics-bystrandify` → `pbmm2 align --preset HIFI` against the sample's own Flye assembly → `ipdSummary --identify m6A,m4C --methylFraction` → motifMaker (SMRT Link 25.3) ([Flusberg et al. 2010](https://doi.org/10.1038/nmeth.1459); [Clark et al. 2012](https://doi.org/10.1093/nar/gkr1146)).

## 3. Observations from the initial profiling (2026-10-19)

Pooled across the 45 `motif.csv` files:

1. **GANTC m6A is in 44/45 genomes**, fraction 0.986 ± 0.010 and mean IPD ratio ≈ 3.9. This is the CcrM motif, which is conserved in Alphaproteobacteria ([Adhikari & Curtis 2016](https://doi.org/10.1093/femsre/fuw023)). It serves as an internal positive control.
2. **MAG154_2 has no GANTC.** It is not in `final_dataset.txt` and has no `flye_asm/MAG154_2_filtered` folder, so it was already excluded upstream. Its methylome is entirely different: fully methylated YACATC, GTTGAG, and CCAGN6CTC/GAGN6CTGG. That fits a non-*S. meliloti* or contaminated sample. **Exclude it.**
3. **The fraction distribution is bimodal.** 38 non-GANTC calls have fraction > 0.9, while most m4C calls are below 0.4: of 320 non-GANTC calls, 198 are ≤ 0.4.
4. **m4C signals are weak.** The median `meanIpdRatio` is 2.72 for m4C and 4.22 for m6A, and the median `meanScore` is 138 vs 239. This matches the known lower kinetic sensitivity of m4C.
5. **The GACTC family looks like a GANTC shadow.** About 15 degenerate variants appear (GACTCA, GACTCAD, GACTCANND, GACTCNKYA, VNGACTCANND, …). All are m4C at fraction ~0.4–0.65, and the called C sits immediately 3′ of the CcrM-methylated A (GA**C**TC). This is the typical pattern of an m6A IPD signal spilling onto the neighbouring base.
6. **The same motif is reported several ways:**
   - **Bipartite motifs listed once per strand,** often without `partnerMotifString` filled in:
     - CTACN5GGC (m6A) / GCCN5GTAG (m4C)
     - CCAGN6CTC / GAGN6CTGG
     - AYCN6TGGA / TCCAN6GRT
     - TCABN6CTGC / GCAGN6VTGA
     - CTGGAG / CTCCAG
   - **m4C "shadows" of strong m6A motifs:** GGTACC (m6A) alongside GGTACCNV / GGTACCGV (m4C); CTCGAG / BCTCGAGV / GACTCGAG.
7. **`centerPos` is 1-based in these files.** Checked against GFF contexts, where the modified base is at index 20 of the 41-mer: the m6A sits on the A at 0-based index 1 of GANTC, while `centerPos` = 2. All position arithmetic below uses `pos0 = centerPos − 1`.

## 4. Pipeline

### Phase 0 — Sample-level gate
Keep sample *s* only if:
- *s* is in `final_dataset.txt` and has an assembly in `flye_asm/`;
- GANTC m6A is called with fraction ≥ 0.9 (positive control);
- median `meanCoverage` ≥ 50× *(tunable; observed range 92–206×, so all current samples pass)*.

Expected result: 44 samples retained (MAG154_2 dropped).

### Phase 1 — Motif-level confidence tiers (motif.csv only)
Assign each motif call to a tier instead of applying one hard cutoff:

| Tier | Rule *(tunable)* | Use |
|---|---|---|
| **High** | fraction ≥ 0.85 and nGenome ≥ 50 | presence/absence matrices |
| **Partial** | 0.40 ≤ fraction < 0.85 and nGenome ≥ 50 | kept and flagged; candidates for phase variation or shadows; site-level check required |
| **Low** | fraction < 0.40, or nGenome < 50, or `modificationType = modified_base` with motif length > 15 | excluded from matrices; listed in catalog |

nGenome ≥ 50 removes motifs whose fraction is estimated from too few sites. Very long, mostly-N motifs (e.g. `TNNNNNNNNNNNNNCNTAGGANNNANC`) are typical motifMaker over-fits.

### Phase 2 — Sequence-level harmonization (motif.csv only)
1. **Normalize:** convert to uppercase IUPAC, trim leading and trailing N, and record `pos0`.
2. **Reverse-complement pairing.** For each genome, pair motif A with motif B when `revcomp(A) == B` after trimming, regardless of `partnerMotifString`. Store the pair as one entity, `A/B`, keeping a modification type and position for each strand.
3. **Canonical key** = lexicographically smaller of (motif, revcomp(motif)). The key is the family unit. Both strands of a bipartite motif share the key, even when they report different modified positions or mod types (e.g. CTACN5GGC m6A + GCCN5GTAG m4C). All of their modified positions are kept.
4. **Families (near-identity, not subsumption).** Align two units on their modified bases, in both orientations, and take the position-wise IUPAC intersection *I*. Merge them if `bits(I) − min(bits(A), bits(B)) ≤ 1` *(tunable)*, where `bits` = Σ log2(4/|set|). This means *I* keeps ≥ 50% of the sites of **both** motifs.
   - **Plus strict subsumption between High-tier units:** with the modified base aligned, every site of one is a site of the other. This joins halves of one system, e.g. BCTCGAGV and CTCGAGTC (= GACTCGAG) under CTCGAG. Both were High in MAG221_1, so they had been counted as two MTases.
   - A sanity check warns if any genome still has two High families that overlap at the modified base. There are none after this fix.
   - *Implementation note:* the first run used plain subsumption for all tiers. Degenerate motifs (e.g. SVVCNK, BNCND) then acted as hubs and chained 126 unrelated m4C variants into one family. The symmetric near-identity rule, with subsumption only for High units, gives 149 canonical motifs → 63 families.
   - The family representative is the unit called High in the most genomes, then present in the most genomes, then the most general (the core of the system).
5. **Candidate shadows (sequence level).** A motif is a candidate shadow when every one of its sites contains a High-tier m6A motif from the same genome, with the modified base 1–5 bp away (e.g. GACTCAD always contains GANTC, with the C adjacent to the m6A). Phase 3 confirms these with called sites.
6. Output: `family_id`, representative, variants, tier counts per family, `phase2_status` ∈ {retain, candidate_shadow, partial_check_site_level, low_exclude}.

### Phase 0–2 results (run 2026-10-19)
- Phase 0: 44/45 samples pass. MAG154_2 fails (`not_in_final_dataset; no_assembly; GANTC_control_failed`).
- Phase 1 (pass samples): 79 High, 82 Partial, 194 Low calls.
- Phase 2: 63 families. 21 retain (High in ≥ 1 genome), 11 candidate_shadow, 8 partial_check_site_level, 23 low_exclude. Family IDs are assigned per run; use the representative motif as the stable name.
  - **Retained:** GANTC (44/44). The rest are mostly strain-specific, fully methylated m6A motifs: ACGGAG (4), CTACN5GGC (3), GGTACC (3), CTCGAG (2), and 1-genome motifs such as CTCCAG, AYCN6TGGA, GCAGN6VTGA, TCGAN8TCGA. There are also a few m4C motifs: ACCGGT, CACCGC, CCGCTCG, TCGWCGA, CCCN5GGG.
  - **Candidate shadows:** GACTCAD and GACTCNKYA families (43 and 32 genomes; GANTC ±1); BNGGTACC / BCGGTACC (GGTACC +1/+2). Plus 1-genome m4C shadows of GCTTGAC, GTTCGAC, CCACAY, GCATYAC and TCABN6CTGC.
  - **To resolve at site level:** two m4C families are called in 43/44 genomes at low fraction:
    - GNCTGVKCA (27 variants, median fraction 0.26);
    - GRCYSANYRD (median 0.28).
    
    They are either a real, weakly detected m4C system or a recurrent artifact. These matter most for the existing clustering, because they appear almost everywhere.
- `genome_family_matrix.tsv`: 44 genomes × 21 families (High-tier calls only).

### Phase 3 — Site-level validation (per genome; needs GFF + assembly)
For each harmonized motif in each genome:
1. **Enumerate all motif sites** on both strands of the exact assembly used for calling (`flye_asm/<s>_filtered/<s>_filtered_asm.fasta`; contig names match the GFF).
2. **Join to detected calls** in `basemods.gff` at `pos0` and strand. This gives the fraction of sites detected per contig and replicon. The motif-level fraction from motifMaker is genome-wide, so this step recovers replicon-level signal.
3. **Shadow test, two parts.**

   **(a) Kinetic footprint (the independent test).** For every High m6A site, take kinetics from `basemods.csv` at offsets −5…+5 on both strands, labelled by the base at that position. Positions that are themselves High m6A sites are excluded. Output the median IPD, the share with IPD ≥ 2, and the share called m4C, by offset × strand × base.
   - The key comparison is the N of GANTC when N = C versus N = A/G/T.
   - IPD elevated only at C, with A/G/T flat, supports either a real m4C or a base-specific neighbour effect of the m6A.
   - IPD at C no different from A/G/T means the GACTC calls are spillover.
   - Fully separating "real m4C" from "neighbour effect" would need an unmethylated (WGA) control, which we don't have.

   **(b) Called-site offset test.** For each Partial/Low motif M:
   - Compute the share of M's called sites that lie within ±5 bp of a site of a High-tier m6A motif in the same genome.
   - Compute the most frequent offset between the two.
   
   If ≥ 80% of M's called sites sit at a single fixed offset from the m6A site *(tunable)*, flag M as `shadow_of=<m6A motif>`. **Caveat:** for motifs that Phase 2 already flagged as sequence-contained (GACTC* in GANTC, BNGGTACC in GGTACC), this test scores 1.0 by construction. It is only informative for motifs that are *not* contained. For contained ones, a positive result is only *consistent with* a shadow; the decision rests on (a).
   - The offset is defined as (M position − m6A position) in M's strand orientation, together with `same` or `opposite` strand.
   - For palindromic m6A motifs, two offsets are equally valid. For GACTC vs GANTC these are +1 on the same strand and −1 on the opposite strand, and the output reports one of them.
4. **Nesting check at site level.** For each family, confirm that the representative motif explains ≥ 90% of the called sites of its variants. If not, split the family.

### Phase 4 — Per-site kinetics, including undetected sites (needs per-base CSV)
`basemods.gff` lists **only detected** bases. To tell "site not methylated" apart from "site not covered", use `basemods.csv` (one row per position and strand) at every motif site from Phase 3, extracting `coverage`, `ipdRatio`, `score` and `frac`.
- **Denominator QC:** count sites with coverage ≥ 25× *(tunable)*.
- **Per-replicon methylated fraction:** chromosome, pSymA, pSymB and accessory plasmids, with Wilson 95% CI.
- **Unmethylated-site list** for High-tier motifs (IPD ratio < 1.5 at coverage ≥ 25×). This is the input for the later "site-level / partial methylation" analysis (promoters, GANTC protection).

### Phase 3–4 implementation notes
- Coordinates: GFF positions and `basemods.csv` `tpl` are both 1-based, and CSV strand 0 = `+`, 1 = `−`. This was verified on Liana_087_2: `tpl` 1, strand 1, score 39 equals GFF pos 1 `−` score 39. In the smoke test, the scores in the CSV matched the GFF scores at the same sites.
- **`detected` is not calibrated.** It means *any* `basemods.gff` row at the site (any type, any score), which is looser than motifMaker's per-motif `fraction`. On contig_4, GRCYSANYRD is "90% detected" against motifMaker's genome-wide 0.23, and BVVCNK sits at 12% with median IPD 1.07, i.e. background. Phase 5 must calibrate before using it:
  1. apply the score cutoff `pbmotifmaker find` used (it was run with default parameters, per `methyl_call_pacbio_hifi.sh`; check the default in the SMRT Link 25.3 docs), using the `call_score` column;
  2. check that the genome-wide site-level fraction then reproduces motifMaker `fraction` for GANTC and the High motifs;
  3. estimate a per-genome background rate (GFF calls on A/C outside any motif ÷ such bases) as the null for contig-level presence.
- Smoke test (Liana_087_2, contig_4, 75 kb, 6 motifs):
  - 16 s wall, 0.6 GB RAM; the full 1.1 GB CSV is streamed in 5 M-row chunks.
  - **m6A footprint:** at the N of GANTC (offset +1, same strand), median IPD is 1.50 when N = C (n = 55) versus 1.01–1.14 when N = A/G/T, and 11% of the C sites are called m4C. This is a weak, C-specific signal. It is too few sites to conclude from; wait for the genome-wide numbers.
  - GANTC: 267/268 covered sites detected (0.996).
  - GACTCAD / GGACTCBNCD: 100% of called sites at a fixed 1 bp offset from GANTC (shadow = True). This is expected by construction (see 3b).
  - GRCYSANYRD / VSGNCYGVNCA: not near m6A, so shadow = False. About 90% of their sites on this contig are detected, but median IPD is only 1.7–1.9, so these need the genome-wide view.
- Sites spanning the origin of circular contigs are not enumerated (≤ motif length per contig).
- Per-sample outputs in `methyl_call/motif_QC/motif_sites/`:
  - `<s>.sites.tsv.gz`
  - `<s>.motif_replicon_summary.tsv`
  - `<s>.shadow_test.tsv`
  - `<s>.family_overlap.tsv`
  - `<s>.m6A_footprint.tsv`
  
  Output is `.tsv.gz` rather than parquet, because base conda has no pyarrow or polars.

### Phase 5 — Outputs (replace the current matrix)
Written to `methyl_call/motif_QC/`. Files marked ✔ already exist.

| File | Content |
|---|---|
| ✔ `sample_qc.tsv`, `samples_pass.txt` | Phase 0 gate result per sample; array sample list |
| ✔ `motif_calls_annotated.tsv` | every motif.csv row + tier, canonical key, rc partner, family_id, shadow candidate |
| ✔ `motif_catalog.tsv` | one row per family: representative, variants, mod type(s), positions, n genomes per tier, phase2_status |
| ✔ `genome_family_matrix.tsv` | genomes × families (High tier = 1) |
| ✔ `figures/fig1_fraction_tiers.*`, `fig2_family_tier_heatmap.*` | fraction by tier; best tier per genome × family |
| `motif_sites/<s>.*` (HPC) | per-site table and per-sample summaries (see above) |
| ✔ `phase5/*` | calibration, background, footprint, family status, contig × family matrices (see Phase 5 results) |
| ✔ `phase5/figures/fig3–fig5` | calibration + background; GANTC N-position footprint; contig × family fraction heatmap |

`methylation_correlation_analysis.py` and `methylation_dendrogram_tanglegram.py` should then be re-run against `contig_family_matrix.tsv`, and the results compared with the current figures.

### Phase 5 results (`motif_qc_aggregate.py`, run 2026-10-19)
**Calibration.** The GFF score cutoff that reproduces motifMaker `nDetected` is **t\* = 101**, i.e. `pbmotifmaker find` counts sites with score > 100.
- The minimum is sharp: the median |log ratio| is 0.002 at 101, against 0.012 at 100 and 0.020 at 102.
- At t\*, the site-level fraction matches motifMaker `fraction` with r = 1.000. 89% of sample × motif pairs are within 2% of `nDetected`.
- Enumerated sites equal `nGenome` in every pair.
- The only larger deviations are highly degenerate Low motifs (SVGCVK, BNCN6D: up to −27%).
- All site-level numbers below use score ≥ 101 among sites with coverage ≥ 25×.

**Background.** GFF calls with score ≥ 101 on A/C outside every enumerated motif site occur at 4.7 × 10⁻⁴ per base (median; range 2.6 × 10⁻⁴ to 1.3 × 10⁻³). Every motif motifMaker reports is ≥ 100× above this, so enrichment over background does **not** discriminate weak motifs. It is reported but not used as a filter.

**GANTC N-position footprint (all 44 genomes).** At offset +1 on the same strand, median IPD is:

| N in GANTC | Median IPD (range across genomes) |
|---|---|
| C | 1.61 (1.52–1.73) |
| T | 1.17 |
| A | 1.05 |
| G | 1.00 |

- C exceeds max(A, G, T) in 44/44 genomes, by 0.40–0.51.
- 17% (12–25%) of these C sites are called m4C. No A, G or T sites are.
- So the GACTC* m4C calls come from a reproducible, C-specific kinetic signal tied to every GANTC site. That fits a neighbour effect of CcrM m6A on the adjacent C better than an independent m4C system, which would also have to be present in all 44 strains.
- This can't be fully separated without an unmethylated (WGA) control.
- Either way it carries no strain information, so these families are excluded from the matrices.

**Final family status** (`phase5/family_status.tsv`, 63 families):

| Status | n | Rule | Examples |
|---|---|---|---|
| retain | 21 | High tier in ≥ 1 genome | GANTC (44), ACGGAG (4), CTACN5GGC (3), GGTACC (3), CTCGAG (2), 16 single-genome systems |
| m6A_linked_excluded | 12 | Phase 2 candidate shadow, or ≥ 80% of called sites within 5 bp of a High m6A site | GACTC* (43/32/3 genomes), BNGGTACC, BCGGTACC, single-genome shadows of GCTTGAC, GTCGAAC, CCACAY, GCATYAC, TCABN6CTGC |
| weak_methylation_flag | 20 | calibrated fraction ≥ 0.20 *(tunable)*, not m6A-linked | GNCTGVKCA (43 genomes, 0.26), GRCYSANYRD (43, 0.27), TGCCAGGYB (5), CRTTCRGYC (5), CCTGCAGGS (3) |
| background_excluded | 10 | calibrated fraction < 0.20 | MNGBBS, MBGCBS, BNCND (≈ 0.005), BCTCYGVNCA (0.18) |

The weak families are kept out of the main matrix (separate `_weak` files). Whether they reflect real partial m4C or sequence-context artifacts of the IPD model is still open. The next test is per-site IPD bimodality: a real partial methylation should split sites into methylated and unmethylated groups, whereas an artifact would shift all sites slightly.

**Contig × family matrix** (`phase5/contig_family_fraction.tsv` / `_presence.tsv`, 174 contigs × 21 retained families).
- Presence = calibrated fraction ≥ 0.5 with ≥ 10 covered sites.
- NaN = the family was not scanned in that genome, or the contig has too few sites.
- **Within a genome, every retained motif is methylated at similar levels across chromosome, pSymA, pSymB and accessory plasmids.** Contig fractions are ≥ 0.87, except for three accessory contigs that are worth a follow-up:
  - Liana_075_3 contig_1 (74 kb): GANTC 0.75
  - MAG514_1 contig_5 (36 kb): GANTC 0.65
  - MAG221_1 contig_1 (237 kb): CTCGAG 0.79

  Candidate explanations: lower plasmid copy number or coverage, replication timing, or local loss of protection.

**Limitation that matters for the plasmid questions.** `motif_sites_qc.py` scans only the motifs motifMaker called in each genome. So the matrix cannot show an accessory plasmid that carries a methylation pattern its host lacks: a plasmid-borne MTase whose motif is too rare genome-wide for motifMaker to call. Re-running the worker with every retained family scanned in all 44 genomes would close that gap. It is cheap, at about 1.5 min per task.
- **Implemented 2026-10-19:**
  - `motif_sites_qc.py --scan retained` scans every strand-specific High-tier unit of the 21 retained families (27 units) in every genome. Units not called by motifMaker in a genome get `tier = not_called`.
  - Submit with `sbatch --export=ALL,SCAN=retained motif_sites_qc_slurm.sh`; output goes to `motif_sites_retained/`.
  - Aggregate with `motif_qc_aggregate.py --sites-dir <qc-dir>/motif_sites_retained --tstar 101 --outdir <qc-dir>/phase5_retained`. The cutoff is fixed at 101 because the called-mode calibration already established it; family status is computed from called genomes only.
  - Smoke test (Liana_087_2 contig_4): GANTC 0.996. All 25 families not called in that genome had uncalibrated detection of 0–0.20, as expected for an absent MTase.

### Phase 5b — Retained-family rescan (job 5446487, 2026-10-19/20)
- **Run:** `motif_sites_qc.py --scan retained`, 44/44 tasks, ≤ 54 s and ≤ 2.2 GB each, aggregated into `phase5_retained/` with `--tstar 101`. Families that weren't scanned are labelled `not_scanned`.
- **Coverage:** 3,636 of the 3,654 contig × family cells now have a measured fraction.
- **The result is fully bimodal:**
  - Where a family **was not** called by motifMaker in that genome, its fraction is ≤ 0.097 on every contig (median 0.001; 3,351 cells), accessory plasmids included.
  - Where it **was** called, every contig is ≥ 0.65 (median 0.988; 285 cells).
- **So no accessory plasmid carries methylation its host lacks.** Every plasmid's profile follows the host's methyltransferases (MTases). Within this set of 21 motif families, there is no sign of plasmid-encoded MTases that are active only on the plasmid.

### Phase 7 — Host / plasmid profile comparisons (`methylation_profile_tests.py`)
Outputs are in `methyl_call/motif_QC/profile_tests/`: `results_summary.tsv`, per-test tables, and `figures/`.

**Profile.** Calibrated fractions of the 20 non-GANTC retained families, using contigs with ≥ 10 covered sites.
- GANTC is examined separately, because it is about 1 on every contig.
- For distances, fractions below 0.5 are set to 0. Background values (≤ 0.1) would otherwise scatter the genomes that are identical in practice.
- Distance M = mean absolute difference.

**Q1 — Plasmid vs its host** (42 plasmids)
- **Families the host methylates:** plasmid − host-chromosome difference has median −0.001 (n = 69 plasmid × family).
- **Families the host doesn't methylate:** the maximum plasmid fraction is 0.097, and 0 of 795 are ≥ 0.5.
- **Own-host distance:** median 0.
- **The own-host vs shuffled-host permutation** gives p = 0.0005. That mainly reflects how diverse the hosts are. Twenty hosts are GANTC-only and tie with each other, so own-host rank percentiles can reach about 0.33 by ties alone.
- **Outliers** (|difference| > 0.10) are all **reduced** methylation:

| Plasmid | Motif | Plasmid fraction | Host fraction | Coverage vs chromosome |
|---|---|---|---|---|
| contig_5_MAG514_1 (36 kb) | GANTC | 0.65 | 0.98 | 0.56× |
| contig_1_Liana_075_3 (74 kb) | GANTC | 0.75 | 0.99 | 0.54× |
| contig_4_Liana_052_1 (74 kb) | GANTC | 0.82 | 0.98 | 0.68× |
| contig_3_Liana_056_3 (74 kb) | GANTC | 0.89 | 0.99 | 0.71× |
| contig_1_MAG221_1 (237 kb) | CTCGAG | 0.70 | 0.81 | 1.07× |

- **The four GANTC cases are small plasmids below chromosome coverage, not high-copy ones.** Three are the mobmess cluster-166 plasmid in near-clonal Liana hosts; the same plasmid in Liana_087_2 is at 0.996.
- A hypothesis to test: CcrM methylates GANTC only at a specific point in the cell cycle. So a plasmid that replicates outside that window, or is present in only part of the population, could carry more hemimethylated GANTC sites. Strand-specific GANTC calls at these sites would test it.

**Q2 — Plasmid relatedness vs methylation** (41 plasmids in wGRR; contig_2_MAG245_2 is not in the wGRR table)
- **Pairs:** 809 plasmid pairs from different hosts. The 11 same-host pairs are excluded, because M = H = 0 by construction.
- **Distances:**
  - P = 1 − wGRR, using the larger value of the two directions; pairs with no wGRR entry get P = 1.
  - H = 1 − host ANI/100, from fastANI.
- **Statistics:** Spearman Mantel and partial Mantel. Permutations (9,999) shuffle the host methylation profiles among hosts, because M depends only on the two hosts.

| Test | r | p |
|---|---|---|
| r(M, P) | −0.066 | 0.75 |
| r(M, H) | −0.060 | 0.70 |
| r(M, P \| H) | −0.061 | 0.72 |
| r(M, P), near-clonal hosts collapsed (Liana_052_1/056_3/075_3 at ANI ≥ 99.99; 37 plasmids, 657 pairs) | −0.129 | 0.89 |
| r(M, P \| H), same collapse | −0.121 | 0.87 |
| PERMANOVA M ~ MOB-suite primary cluster (6 clusters with ≥ 2 hosts, one plasmid per host per cluster, n = 34) | pseudo-F 1.38, R² 0.20 | 0.16 |

**No evidence that related plasmids live in hosts with similar MTase repertoires.** Power is limited: 20 hosts share the GANTC-only profile, and most other families occur in a single genome.

**Q3 — Genomes with (32) vs without (12) accessory plasmids**
- MAG245_2 counts as plasmid-carrying: its contig_2 is a 166 kb circular contig at 176×, although it is missing from the plasmid CSV and from wGRR.

| Test | Result | p |
|---|---|---|
| Fisher 2×2 | Genomes with an MTase system beyond GANTC: 8/12 without plasmids (67%) vs 16/32 with plasmids (50%); OR 0.5 | 0.50 |
| Mann–Whitney on the number of extra families | median 1 without, 0.5 with | 0.65 |
| PERMANOVA on host profiles | R² 0.042 | 0.08 |
| PERMDISP | — | 0.74 |

The direction is consistent with an RM barrier, with plasmid-free genomes slightly more often carrying extra systems, but nothing is significant at n = 44.

**Open threads**
1. Motif depletion (observed/expected sites per plasmid) is the remaining place where a plasmid-intrinsic signal could appear.
2. Strand-specific GANTC analysis of the four under-methylated small plasmids.
3. Why contig_2_MAG245_2 is absent from the plasmid tables.

### Phase 6 (next) — Motif ↔ MTase linkage
Correlate family presence/absence across the 44 genomes with MTase genes from rmsFinder and DefenseFinder (RM type I–III, orphan MTases), and check predicted specificities against REBASE ([Roberts et al. 2023](https://doi.org/10.1093/nar/gkac975)). This is not part of the current scope.

## 5. Compute assessment: local (maple) vs HPC

Observed: `maple.sgc.loc` has 48 cores and 377 GB RAM, but it is a shared interactive node.

| Phase | Data read | Estimated cost | Where |
|---|---|---|---|
| 0–2 | 45 small CSVs (364 rows) | seconds, <1 GB RAM | **Local** |
| 3 | 1.4 GB GFF + 44 × ~7 Mb assemblies; regex scan of ~3–13 motifs per genome | ~1–3 min/genome single-core; ~1–2 h serial | **HPC array** (local is possible but would occupy maple) |
| 4 | 14 GB per-base CSV (~15 M rows/genome) | full CSV pass took ~15 s in the smoke test; whole-genome RAM is driven by the number of motif sites (up to ~3.6 M in MAG373_3) and is **not yet measured**. The 16 GB / 2 h request is a guess to be checked on the test task | **HPC array** |
| 5 aggregation + figures | 225 MB site tables + 1.4 GB GFF (background) | 3 min, 1.4 GB (measured) | **Local**; an array is not worth it because the cutoff is fitted across all genomes |
| *(not planned)* re-run ipdSummary/motifMaker | 219 GB aligned BAMs | days of CPU | HPC only |

Phase 3 and 4 timings are estimates and still need to be measured on one genome.

**Recommendation:** run Phases 0–2 and Phase 5 locally. Run Phases 3 and 4 together as **one SLURM array job with 44 tasks, one per genome**. Each task reads its own GFF, CSV and assembly, so tasks are independent and no task has to load all 14 GB. Before submitting the array, test with one genome (e.g. `--array=0`) to measure time and memory.

The job script is `motif_sites_qc_slurm.sh`. It requests 2 CPUs, 16 GB and 2 h per task, with `--array=0-43` indexed over `methyl_call/motif_QC/samples_pass.txt` (the 44 pass samples, not the 256-entry `final_dataset.txt`), and uses the base conda env. Submission stays with the user. Submit from the project root so that `slurm_logs/` resolves:

```bash
cd /n/projects/jp2992/BacPanMeth
sbatch --array=28 motif_sites_qc_slurm.sh  # test on MAG373_3 (heaviest: ~3.6 M motif sites); check time/RAM with sacct
sbatch motif_sites_qc_slurm.sh             # all 44
```

## 6. References
- Flusberg BA et al. (2010) Direct detection of DNA methylation during single-molecule, real-time sequencing. *Nat Methods*. https://doi.org/10.1038/nmeth.1459
- Clark TA et al. (2012) Characterization of DNA methyltransferase specificities using single-molecule, real-time DNA sequencing. *Nucleic Acids Res*. https://doi.org/10.1093/nar/gkr1146
- Adhikari S, Curtis PD (2016) DNA methyltransferases and epigenetic regulation in bacteria. *FEMS Microbiol Rev*. https://doi.org/10.1093/femsre/fuw023
- Roberts RJ et al. (2023) REBASE: a database for DNA restriction and modification: enzymes, genes and genomes. *Nucleic Acids Res*. https://doi.org/10.1093/nar/gkac975
