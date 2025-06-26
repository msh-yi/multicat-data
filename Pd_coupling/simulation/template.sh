#!/bin/bash
#SBATCH --job-name='{job_name}'
#SBATCH --nodes=1 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=12000
#SBATCH --time=10:00:00
#SBATCH --partition=sapphire,shared
#SBATCH --mail-type=END,FAIL

source activate combi

srun python ~/multicat/driver.py -c {config_path} -o {output_file} -m