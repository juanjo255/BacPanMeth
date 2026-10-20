#!/usr/bin/bash
#SBATCH --job-name=motif_sites
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=00:15:00
#SBATCH --array=0-43
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err
#SBATCH --mail-user=jp2992@stowers.org
#SBATCH --mail-type=END,FAIL

## Methylation motif QC — Phases 3-4 (site-level validation + per-site kinetics), one genome per task.
## Prerequisite: Phases 0-2 already run locally (motif_qc_harmonize.py), which writes
##   methyl_call/motif_QC/motif_calls_annotated.tsv and samples_pass.txt (44 samples).
## Submit from the repo root so slurm_logs/ resolves:
##   cd /n/projects/jp2992/BacPanMeth
##   sbatch motif_sites_qc_slurm.sh                                  # motifs called in each genome -> motif_sites/
##   sbatch --export=ALL,SCAN=retained motif_sites_qc_slurm.sh       # all retained families in every genome -> motif_sites_retained/
## Measured (job 5445169, SCAN=called): <= 94 s and <= 3.4 GB per task (heaviest MAG373_3).

set +o noclobber
source /n/projects/jp2992/MOLNG4331/code/log_message_function.sh

## ACTIVATE CONDA (base has pandas, numpy, biopython) ##
. ~/miniforge3/etc/profile.d/conda.sh
conda activate base
## Silence the harmless Biopython warning from a stray pyproject.toml in site-packages
export PYTHONWARNINGS="ignore::UserWarning"

BASE=/n/projects/jp2992/BacPanMeth   # data
REPO=/n/projects/jp2992/BacPanMeth                         # code
samples=($(cat $BASE/methyl_call/motif_QC/samples_pass.txt))
s=${samples[$SLURM_ARRAY_TASK_ID]}
SCAN=${SCAN:-called}                                   # called | retained
OUTDIR=$BASE/methyl_call/motif_QC/motif_sites
[ "$SCAN" = "retained" ] && OUTDIR=${OUTDIR}_retained
log_info "Task $SLURM_ARRAY_TASK_ID -> sample $s (scan: $SCAN)"

python $REPO/motif_sites_qc.py \
    --sample "$s" \
    --qc-dir $BASE/methyl_call/motif_QC \
    --scan "$SCAN" \
    --outdir "$OUTDIR"
