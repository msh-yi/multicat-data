#!/bin/bash
#SBATCH --job-name='11_oxet_0_ls_49'
#SBATCH --nodes=1 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=6000
#SBATCH --time=1:00:00
#SBATCH --partition=sapphire,jacobsen2,shared
#SBATCH --mail-type=END,FAIL

source activate combi

srun python ~/multicat/driver.py -c /n/home10/msak/multicat/batch_inputs_jobs/11_oxet_0/11_oxet_0_ls_49.yaml -o /n/home10/msak/multicat/batch_inputs_jobs/11_oxet_0/out/11_oxet_0_ls_49.out -m