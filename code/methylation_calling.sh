#!/usr/bin/bash
#SBATCH --cpus-per-task=15
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
###SBATCH --partition=bigmem

#SBATCH --mail-user=jp2992@stowers.org
#SBATCH --mail-type=ALL
#SBATCH --job-name=methyl_call
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err


###SBATCH --array=0-46
###SBATCH --partition=gpu
###SBATCH --gres=gpu:a100:3


## Overwrite files with redirection operator (>)
set +o noclobber
cpus=$SLURM_CPUS_PER_TASK

## Import logging messages (log_info, log_warning, log_error)
source /n/projects/jp2992/MOLNG4331/code/log_message_function.sh

ml samtools

## ACTIVATE CONDA ##
. ~/miniforge3/etc/profile.d/conda.sh 


### ALIGN READS ### 

bam_files=(/n/projects/jp2992/Plasmidome_project_MOLNG_4509/rename_bams/*.bam)
task_id=$SLURM_ARRAY_TASK_ID
file=${bam_files[$task_id]}

#####################################################
## GET REQUIRED FILES
## example name: MOLNG4331.A01.bam 
folder_name=$(basename $file | cut -d"." -f 2)
log_info "file $file"
log_info "Task ID $task_id"
log_info "Processing sample: $folder_name"
#####################################################


smrt_cmds="/n/projects/jp2992/MOLNG4331/biocomp_tools/smrtlink/install/smrtlink-release_25.3.0.273777/bundles/smrttools/install/smrttools-release_25.3.0.273777/smrtcmds/bin/"
prefix=$(basename $file .bam)
## In this case I have the assemblies in folders named with the bam file name + _filtered 
## and the assemblies have the same name as the folder
assembly="/n/projects/jp2992/Plasmidome_project_MOLNG_4509/flye_asm/"$prefix"_filtered/"$prefix"_filtered_asm.fasta"
out_dir="/n/projects/jp2992/Plasmidome_project_MOLNG_4509/methyl_call"
source /n/projects/jp2992/MOLNG4331/biocomp_tools/custom_scripts/methyl_call_pacbio_hifi.sh \
    --assembly $assembly --bam $file --outdir $out_dir --smrtPath $smrt_cmds --prefix $prefix --cpus $cpus
