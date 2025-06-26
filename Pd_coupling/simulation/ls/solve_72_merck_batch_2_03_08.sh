#!/bin/bash
#SBATCH --job-name='solve_72_merck_batch_2_03_08'
#SBATCH --nodes=1 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=8000
#SBATCH --time=1:00:00
#SBATCH --partition=sapphire,jacobsen2,shared
#SBATCH --mail-type=END,FAIL

source activate combi

srun python ~/multicat/coop_solver.py -i /n/home10/msak/multicat/batch_inputs_jobs/72_merck_batch_2/ls/72_merck_batch_2_3_8_ls -k 3 -m dimer -s 100 -b
