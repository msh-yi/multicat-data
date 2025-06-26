#!/bin/bash
#SBATCH --job-name='72_merck_batch_2_3_8_ls_38'
#SBATCH --nodes=1 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=12000
#SBATCH --time=10:00:00
#SBATCH --partition=sapphire,shared
#SBATCH --mail-type=END,FAIL

source activate combi

srun python ~/multicat/driver.py -c /n/home10/msak/multicat/batch_inputs_jobs/72_merck_batch_2/72_merck_batch_2_3_8_ls_38.yaml -o /n/home10/msak/multicat/batch_inputs_jobs/72_merck_batch_2/out/72_merck_batch_2_3_8_ls_38.out -m