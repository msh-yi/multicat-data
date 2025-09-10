# Multicat

This repository serves two purposes:
1. Supporting data and code to generate figures for the publication "Accelerating the Discovery of Catalytic Cooperativity." These data and code are contained in the folders `manuscript_figures/`, `oxetane/`, `Pd_coupling/`, and `simulation_testing/`.
2. Code for *`multicat`, a package to generate catalytic cooperativity landscapes and benchmark pooling/deconvolution algorithms*. This package comprises all the python modules in the top-level directory. The remainder of this documentation is about **Multicat**.

---

## Overview

**Multicat** is a research toolkit for **simulating landscapes** that exhibit positive/negative cooperativity and then **stress‑testing pooling–deconvolution (PD) algorithms**.  
While the codebase also contains `rec` (recommendation) and `proc` (processing) pipelines, this README documents the **simulation path (`sim` mode)** that most users need for benchmarking and method‑development studies. Pooling and deconvolution for actual experimental runs can be done on the [Multicat website](https://multicat.onrender.com/). A brief tutorial about the Multicat website can be found below.

---

## Installation

This package was developed and tested on Python 3.9. The package is designed predominantly for use on a Unix system. The simulation workflow is compatible with Slurm and was tested on Harvard's Cannon HPC cluster.

```bash
git clone https://github.com/msh-yi/multicat-data.git
cd multicat-data
conda env create -f environment.yml
conda activate multicat
```
Install time depends on the speed of your conda. We recommend miniforge.

A minimal Conda recipe:

```bash
conda create -n multicat python=3.10 copasi numba pandas pyyaml
conda activate multicat
#pip install basico
```

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


Both **landscape generation** and **simulation** live in a single YAML. Read `template.yaml` for more details. There are many parameters that control the pooling-deconvolution simulation beyond just `k`, `t`, and `r`. Most of the time the other parameters do not need to be modified. It is recommended to prepare two YAML files for every run, one for landscape generation and one for pooling-deconvolution. Specifically, omit the entire `oneshot` block if you only want to generate landscapes.

*Tip: keep list syntax for every hyper-parameter under oneshot, even singletons:*

```yaml
oneshot:
  pool_size: [6]          # good ✓
  num_meet: [2, 3]        # grid - two values
```

---

## Step‑by‑Step Workflow
You can follow along in this minimal example by going to the `demo/` folder. We start with `template_landscape.yml` and `pd_landscape.yml`. See above for a detailed explanation on these configuration files.

### 1️⃣ Generate Landscapes

```bash
python driver.py -c demo/demo_ls.yml -g        # '-g' = generate landscape
```
In this case we only generate one landscape. For batch landscape generation, set the `num_landscapes` parameter to > 1 and the `batch` parameter to `True`.

Outputs:

```
demo/demo_ls_ls_cat_rate.csv   # individual rates
demo/demo_ls_ls_coop_mat.csv   # coop multipliers (+ pos, – neg)
```

### 2️⃣ Ground‑Truth Solver

```bash
python coop_solver.py \
     -i demo/demo_ls_ls_cat_rate.csv                       \        # prefix to *_cat_rate.csv / *_coop_mat.csv
     -k 3                         \        # largest k-set to compute (dimer = 2, trimer = 3, …)
     -m dimer                     \        # landscape mode; must match YAML
     -s 100                       \        # strength of negative coop pairs over positive coop pairs

```

Creates `demo_ls_ls_soln.csv` containing all the singles, doubles, and triples ranked from highest rate to lowest rate. This is essential for us to know what the "correct" most cooperative pair is.

Tip: for larger jobs with large batches of landscapes, this solver is parallelized so submission as a parallel job (e.g. on SLURM) is recommended.

```bash
python coop_solver.py \
     -i path/to/landscapes/<prefix> \        # include prefix to *_cat_rate.csv / *_coop_mat.csv
     -k 3                         \        # largest k-set to compute (dimer = 2, trimer = 3, …)
     -m dimer                     \        # landscape mode; must match YAML
     -s 100                       \        # strength of negative coop pairs over positive coop pairs
     -b                                    # batch mode → parallel

```

### 3️⃣ Simulate Pooling‑Deconvolution

We can now simulate pooling-deconvolution. This uses `demo_pd.yml`, which contains the path to the landscape files we want to work with, as well as various hyperparameters we want to test with.

```bash
python driver.py -c demo/demo_pd.yml -m -o demo/out/demo_pd.out       # '-m' = simulate, '-o': path to output file
```

This generates one log (specified by the `-o` flag), and a CSV (`/demo/out/demo_pd.csv`) when the run is completed. The CSV aggregates sensitivity, efficiency, and various other metrics for each (k,t,r) parameter set. This run should take no more than two minutes on a portable laptop computer. 

### 4️⃣ Scale up with SLURM

It is possible to do Steps 1 to 3 simultaneously with many landscapes. The code is optimized for parallel computing and performs best on a multi-core setup (e.g. a HPC node with > 24 cores).

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

In this case, the file structure will look like:

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

## Reproducing simulation results from the paper

The simulated landscapes and results from pooling-deconvolution are located in the respective folders as listed above. All of the outputs can be reproduced by following the above instructions, including step 4.

## Experimental pooling-deconvolution

The [Multicat website](https://multicat.onrender.com/) provides a simple interface for pooling-deconvolution that is separate from the simulation workflow, although the code backend for the website is included in this package. (See `rec` or `proc` mode in `driver.py` and `oneshot_pooler.py`.) The website contains all the necessary instructions.

## Contact

For questions, please contact [Marcus Sak](mailto:msak@g.harvard.edu).