#!/bin/bash
#SBATCH --job-name='{job_name}'
#SBATCH --nodes=1 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=6000
#SBATCH --time=1:00:00
#SBATCH --partition=sapphire,jacobsen2,shared
#SBATCH --mail-type=END,FAIL

source activate combi

srun python ~/multicat/driver.py -c {config_path} -o {output_file} -m