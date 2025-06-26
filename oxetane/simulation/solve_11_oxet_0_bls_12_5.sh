#!/bin/bash
#SBATCH --job-name='solve_11_oxet_0_bls_12_5'
#SBATCH --nodes=1 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8000
#SBATCH --time=1:00:00
#SBATCH --partition=sapphire,jacobsen2,shared
#SBATCH --mail-type=END,FAIL

source activate combi

srun python ~/multicat/coop_solver.py -i /n/home10/msak/multicat/batch_inputs_jobs/11_oxet_0/11_oxet_0_ls/11_oxet_0_ls -k 3 -m dimer -s 100 -b

