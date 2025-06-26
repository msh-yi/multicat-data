#!/bin/bash
#SBATCH --job-name=solve_50_jun_grid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=8000
#SBATCH --time=1:00:00
#SBATCH --array=0-24
#SBATCH --partition=sapphire,jacobsen2,shared
#SBATCH --mail-type=END,FAIL

source activate combi

# Make a Bash array that holds every “ls” directory you have to solve
dirs=( $(ls -d ./*/ls) )

# Pick the directory that corresponds to this array index
mydir=${dirs[$SLURM_ARRAY_TASK_ID]}

echo "[$SLURM_JOB_ID,$SLURM_ARRAY_TASK_ID]  Solving $mydir"
srun python ../../coop_solver.py -i "$mydir" -k 3 -m dimer -s 100 -b