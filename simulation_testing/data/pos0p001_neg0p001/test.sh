#!/bin/bash
#SBATCH --job-name='test'
#SBATCH --nodes=1 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=10000
#SBATCH --time=1:00:00
#SBATCH --partition=sapphire
#SBATCH --mail-type=END,FAIL

source activate combi

srun python ~/multicat/driver.py -c /n/home10/msak/multicat/batch_inputs_jobs/50_jun_grid/50_jun_grid_00_pos0p001_neg0p001/landscape_pos0p001_neg0p001_ls_0.yaml -o /n/home10/msak/multicat/batch_inputs_jobs/50_jun_grid/50_jun_grid_00_pos0p001_neg0p001/out/grid_00_pos0p001_neg0p001_ls_0.out -m