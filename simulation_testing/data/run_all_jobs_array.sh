#!/bin/bash
#============ Slurm options ========================================
#SBATCH --job-name=multicat_array
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem-per-cpu=15000
#SBATCH --time=30:00:00
#SBATCH --partition=sapphire
#SBATCH --mail-type=END,FAIL
#===================================================================

source activate combi

# ---------------------------------------------------------
# Build a bash array that contains every *per-landscape* YAML
# (we skip the master "os_*.yml" files).
# ---------------------------------------------------------
if [[ -n "$YAML_LIST" && -f "$YAML_LIST" ]]; then
    mapfile -t configs < "$YAML_LIST"
else
    ROOT=./data
    configs=( $(find "$ROOT" -maxdepth 2 -name '*.yaml' ! -name 'os_*.yml' | sort) )
fi

ROOT=./data
configs=( $(find "$ROOT" -maxdepth 2 -name '*.yaml' ! -name 'os_*.yml' | sort) )

CFG=${configs[$SLURM_ARRAY_TASK_ID]}          # pick the right YAML
OUT=$(dirname "$CFG")/out/$(basename "${CFG%.yaml}.out")  # results file (change if you like)

echo "[${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}]  $CFG -> $OUT"

srun ../../driver.py -c "$CFG" -o "$OUT" -m