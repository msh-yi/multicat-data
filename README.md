# Multicat
*A package to generate catalytic cooperativity landscapes and benchmark pooling/deconvolution algorithms*

---

## Overview

**Multicat** is a research toolkit for **simulating landscapes** that exhibit positive/negative cooperativity and then **stress‑testing pooling–deconvolution (PD) algorithms**.  
While the codebase also contains `rec` (recommendation) and `proc` (processing) pipelines, this README exclusively documents the **simulation path (`sim` mode)** that most users need for benchmarking and method‑development studies. Pooling and deconvolution for actual experimental runs can be done on the [Multicat website](https://multicat.onrender.com/).

Use cases include:

* Building ground‑truth datasets for hypothetical landscapes.  
* Comparing PD algorithms across thousands of random landscapes.  
* Generating error models for downstream statistical power analyses.

---

## Key Features

| Feature | Description |
|---------|----------------|
| **Landscape generator** (`pool_cat_sim.py`) | Creates rate constants & cooperativity matrices for hypothetical cooperativity landscapes |
| **Ground‑truth solver** (`coop_solver.py`) | Exhaustively solves for the best k‑set of catalysts. |
| **One‑shot pooler** (`oneshot_pooler.py`) | Implements pooling-deconvolution workflow. |
| **High‑throughput driver** (`driver.py`) | Orchestrates generation or simulation with a single YAML file. |
| **Cluster batch submission** (`batcher.py`) | Auto‑produces & submits SLURM scripts, injecting config paths and job names into a user‑supplied template. |
| **Result parser** (`process_output.py`) | Converts noisy driver logs into tidy CSV files for downstream plotting. |

---

## Installation

```bash
git clone https://github.com/your-org/multicat.git
cd multicat
pip install -r requirements.txt      # Pure‑Python deps
```

> **Additional requirements**  
> * **SLURM** – workload manager (`sbatch`, `srun`).  
> * A C/C++ build chain (for Numba’s JIT fallback if LLVM is missing). 
> * (Optional) **COPASI + Basico** – deterministic ODE solver backend. The default uses scipy and Numba but solving with Basico is also possible.

A minimal Conda recipe:

```bash
conda create -n multicat python=3.10 copasi numba pandas pyyaml
conda activate multicat
pip install basico
```

---

## Repository Layout (sim‑relevant files)

```
multicat/
├── batcher.py          # generate + submit SLURM scripts
├── coop_solver.py      # brute‑force solution generator
├── driver.py           # main entrypoint (generate / simulate)
├── oneshot_pooler.py   # PD algorithms
├── pool_cat_sim.py     # rate & cooperativity generator
├── process_output.py   # log → CSV
├── template.yml        # example YAML (edit me!)
└── template.sh         # SBATCH skeleton (edit me!)
```

---

## Anatomy of `template.yml`

`template.yml` ships with two ready-to-uncomment scenarios:

| Scenario | Key                      | What it does                                                                                            |
| -------- | ------------------------ | ------------------------------------------------------------------------------------------------------- |
| **1**    | `simul` *(single block)* | **Generate** `num_landscapes` random landscapes. Sets statistical knobs (`N`, `p_pos`, `mean_rate`, …). |
| **2**    | `simul` + `oneshot`      | **Load** an existing landscape (`landscape:`) **and** grid-search PD hyper-parameters under `oneshot:`. |


Both **landscape generation** and **simulation** live in a single YAML. Read `template.yaml` for more details. There are many parameters that control the pooling-deconvolution simulation beyond just `k`, `t`, and `r`. Most of the time the other parameters do not need to be modified. It is recommended to prepare two YAML files for every run, one for landscape generation and one for pooling-deconvolution.

Tips:

1. Omit the entire `oneshot` block if you only want to generate landscapes.
2. Tip: keep list syntax for every hyper-parameter under oneshot, even singletons:

```yaml
oneshot:
  pool_size: [6]          # good ✓
  num_meet: [2, 3]        # grid - two values
```

---

## Step‑by‑Step Workflow

### 1️⃣ Generate Landscapes

```bash
python driver.py -c landscape_template.yml -g        # '-g' = generate landscape
```

Outputs (for each seed):

```
<landscape_name>_cat_rate.csv   # individual rates
<landscape_name>_coop_mat.csv   # coop multipliers (+ pos, – neg)
```

### 2️⃣ Ground‑Truth Solver

```bash
srun python coop_solver.py \
     -i toy                       \        # prefix to *_cat_rate.csv / *_coop_mat.csv
     -k 3                         \        # largest k-set to compute (dimer = 2, trimer = 3, …)
     -m dimer                     \        # landscape mode; must match YAML
     -s 100                       \        # strength of negative coop pairs over positive coop pairs
     -b                                    # batch mode → parallel

```

Creates `<landscape_name>_soln.csv` containing the exact best k‑set and its yield. This solver is parallelized so submission as a parallel job (e.g. on SLURM) is recommended for large (N > 30) landscapes.

### 3️⃣ Simulate Pooling‑Deconvolution

```bash
python driver.py -c pd_template.yml -m        # '-m' = simulate
```

Generates one log per replicate (`out_seedXXXX.out`).

### 4️⃣ Parse Results

```bash
python process_output.py out_seed*.txt -o results.csv
```

The CSV contains accuracy, efficiency, and various other metrics for each (k,t,r) parameter set.

### 5️⃣ Scale up with SLURM

1. **Edit** `template.sh`  
   Key placeholders: `{job_name}`, `{config_path}`, `{output_file}`, as well as SLURM directives.

2. **Run**

```bash
python batcher.py \
       -c template.yml \
       -t template.sh \
       -s                \          # actually submit; omit to just generate .sbatch files
       --seed 42                    # starting RNG seed ➜ batcher increments per job
```

For each landscape, `batcher.py` will submit one job with `sbatch`. For a typical 50-landscape run, 50 slurm jobs are expected, resulting in 50 out files.

---

## Result Directory Example

```
batch_job/
├── landscapes/
│   ├── 0001_cat_rate.csv
│   ├── 0001_coop_mat.csv
│   └── 0001_soln.csv
├── out/
│   ├── 0001.out
└── └── 0001.out.csv
```


## FAQ

**Q: Can I import my own landscape?**  
Yes—set `simul: landscape_mode: file` and supply `landscape:` pointing to an existing `*_cat_rate.csv` prefix.

**Q: Does Multicat support GPU acceleration?**  
Not currently; Numba JIT covers the CPU hot‑spots. Pull requests welcome!
