#!/bin/bash
#SBATCH --job-name='{job_name}'
#SBATCH --nodes=1 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem-per-cpu=15000
#SBATCH --time=30:00:00
#SBATCH --partition=sapphire
#SBATCH --mail-type=END,FAIL

source activate combi

srun python ~/multicat/driver.py -c {config_path} -o {output_path} -m