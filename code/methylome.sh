#!/usr/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00

#SBATCH --mail-user=jp2992@stowers.org
#SBATCH --mail-type=ALL
#SBATCH --job-name=rmsfinder
#SBATCH --output=slurm_logs/%x_%a.out
#SBATCH --error=slurm_logs/%x_%a.err

#SBATCH --array=0-43



## Overwrite files with redirection operator (>)
set +o noclobber
cpus=$SLURM_CPUS_PER_TASK
if [ -z $cpus ];then
cpus=8
fi

#cpus=32
source /n/projects/jp2992/MOLNG4331/code/log_message_function.sh

## Import modules
ml csvtk seqkit samtools

## ACTIVATE CONDA ##
. ~/miniforge3/etc/profile.d/conda.sh



################################################
##### Annotate defense systems
################################################
run_defensefinder (){
  conda activate defensefinder
  dirs=($(ls -d /n/projects/jp2992/Plasmidome_project_MOLNG_4509/all_genomes_annotations/*))
  task_id=$SLURM_ARRAY_TASK_ID
  dir="${dirs[$task_id]}"
  contigs=("Chromosome" "pSymA" "pSymB")
  for i in ${contigs[@]} 
  do
    fasta=$dir"/*_$(basename $dir)_${i}.faa"
    out_dir="/n/projects/jp2992/Plasmidome_project_MOLNG_4509/defensefinder_all_genomes/$(basename $fasta)_defensefinder"
    mkdir $out_dir
    defense-finder run --workers $cpus --db-type ordered_replicon --coverage 0.4 -o $out_dir --antidefensefinder $fasta
  done 

}




############################################################
##### Predict type II restriction system
############################################################
run_rmsfinder(){

  conda activate rmsfinder
  task_id=$SLURM_ARRAY_TASK_ID

  paths_to_genbank=($(cat "/n/projects/jp2992/Plasmidome_project_MOLNG_4509/flye_asm/path_to_genbank.txt"))

  genbank_file="${paths_to_genbank[$task_id]}"

  rmsFinder.py --mode RMS --db gold \
              --output "/n/projects/jp2992/Plasmidome_project_MOLNG_4509/flye_asm/type_2_RM/$(basename $genbank_file .gbff)" \
              --genbank  $genbank_file 

}


### RUN HERE

run_defensefinder