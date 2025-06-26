#!/usr/bin/env python

"""
batcher.py: Generate SLURM job submission scripts for batch PD simulations across multiple landscapes.

This script automates the creation and submission of SLURM job scripts for landscape analysis.
It processes multiple landscape files, creates job-specific configuration files with unique seeds,
and either submits the jobs to SLURM or creates the job scripts for manual submission.

Author: Marcus Sak
Date: 3/18/25
"""

import argparse
import glob
import numpy as np
import os
import re
import subprocess
import sys
import time
import warnings
import yaml

warnings.filterwarnings('ignore')

def read_template(template_path):
    """
    Read contents of a SLURM script template file (usually template.sh).

    Args:
        template_path (str): Path to the template file

    Returns:
        str: Contents of the template file

    Raises:
        FileNotFoundError: If template file doesn't exist
    """
    try:
        with open(template_path, 'r') as f:
            return f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Template file not found: {template_path}")
    except PermissionError:
        raise PermissionError(f"Permission denied: {template_path}")

def create_slurm_script(template, job_name, config_path, output_path):
    """
    Create a SLURM script by populating the template with specific job parameters.

    Args:
        template (str): SLURM script template string
        job_name (str): Name of the job
        config_path (str): Path to the yaml configuration file containing all the PD parameters
        output_path (str): Path where job output should be written

    Returns:
        str: Formatted SLURM script ready for submission
    """
    return template.format(job_name=job_name, config_path=config_path, output_path=output_path)

def natural_sort_key(s):
    """
    Create a key for natural sorting of strings containing numbers.
    
    This function splits strings into numeric and non-numeric parts for proper
    natural sorting (e.g., 'file2' comes before 'file10').

    Args:
        s (str): String to be processed for sorting

    Returns:
        list: List of alternating string and integer components for sorting

    Example:
        >>> sorted(['file2', 'file10'], key=natural_sort_key)
        ['file2', 'file10']
    """

    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()    
    parser.add_argument("-c", "--config", help="path of yaml config file", required=True)
    parser.add_argument("-t", "--template", help="path to SLURM script template (default: template.sh in config directory)", default=None)
    parser.add_argument("-s", "--submit", help="submit SLURM jobs", action="store_true")
    parser.add_argument("--seed", type=int, default=1234, help="master seed for generating job-specific seeds")
    args = parser.parse_args()

    # Get absolute path and directories
    config_abs_path = os.path.abspath(args.config)
    config_dir = os.path.dirname(config_abs_path)
    
    # Define and check landscape directory
    landscape_dir = os.path.join(config_dir, "ls")
    if not os.path.exists(landscape_dir):
        print(f"Error: Required directory '{landscape_dir}' does not exist!")
        print(f"Please create a 'landscapes' directory in {config_dir} with your landscape files.")
        sys.exit(1)
    
    # Define and create output directory if needed
    out_dir = os.path.join(config_dir, "out")
    os.makedirs(out_dir, exist_ok=True)
    
    # Handle template file
    if args.template:
        template_path = os.path.abspath(args.template)
    else:
        template_path = os.path.join(config_dir, "template.sh")
    
    # Check if template exists
    if not os.path.exists(template_path):
        print(f"Error: Template file not found: {template_path}")
        print("Please provide a valid template file with -t/--template or place 'template.sh' in the directory containing your config file.")
        sys.exit(1)

    # Load the config file
    try:
        with open(config_abs_path, 'r') as stream:
            config = yaml.safe_load(stream)
    except yaml.YAMLError as e:
        print(f"Error parsing config file: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"Config file not found: {config_abs_path}")
        sys.exit(1)

    # Read the SLURM script template
    template = read_template(template_path)

    # Use glob to find all relevant files (only the part without prefix)
    all_files = glob.glob(os.path.join(landscape_dir, "*_cat_rate.csv"))

    if not all_files:
        print(f"No landscape files found in {landscape_dir}")
        print("Landscape files should have the format: *_cat_rate.csv")
        sys.exit(1)

    # Extract indices from filenames
    indices = []
    for f in all_files:
        match = re.search(r'(\d+)_cat_rate\.csv$', os.path.basename(f))
        if match:
            indices.append(match.group(1))

    if not indices:
        print(f"No properly formatted landscape files found in {landscape_dir}")
        print("Landscape files should have the format: *_<number>_cat_rate.csv")
        sys.exit(1)

    # Sort indices using natural sort
    sorted_indices = sorted(indices, key=natural_sort_key)

    # Set up the RNG for consistent seed generation
    master_rng = np.random.RandomState(args.seed)
    job_seeds = master_rng.randint(1, 1000000, size=len(sorted_indices))

    # Get landscape prefix from the first file (remove the index and suffix)
    first_file = all_files[0]
    landscape_prefix = re.sub(r'_\d+_cat_rate\.csv$', '', os.path.basename(first_file))
    landscape_prefix_path = os.path.join(landscape_dir, landscape_prefix)

    print(f"Found {len(sorted_indices)} landscape files with prefix: {landscape_prefix_path}")

    # Process each index
    for i, index in enumerate(sorted_indices):
        indexed_ls_name = f'{landscape_prefix_path}_{index}'
        
        # Verify both required files exist for this landscape
        rate_file = f"{indexed_ls_name}_cat_rate.csv"
        coop_file = f"{indexed_ls_name}_coop_mat.csv"
        
        if not os.path.exists(rate_file) or not os.path.exists(coop_file):
            print(f"Warning: Missing required files for landscape {indexed_ls_name}, skipping...")
            continue
            
        # Create a job-specific config file
        job_config = config.copy()
        job_config['simul']['landscape'] = indexed_ls_name
        job_name = f"{os.path.basename(indexed_ls_name)}"
        job_config_path = os.path.join(config_dir, f"{job_name}.yaml")
        
        # Generate a new random seed for each job
        job_config['rng']['seed'] = int(job_seeds[i])

        with open(job_config_path, 'w') as f:
            yaml.dump(job_config, f, default_flow_style=False, sort_keys=False)

        # Create the SLURM script
        output_path = os.path.join(out_dir, f"{job_name}.out")
        slurm_script = create_slurm_script(template, job_name, job_config_path, output_path)
        slurm_script_path = os.path.join(config_dir, f"{job_name}.sh")

        with open(slurm_script_path, 'w') as f:
            f.write(slurm_script)

        if args.submit:
            # Submit the job
            subprocess.run(['sbatch', slurm_script_path])
            print(f"Submitted job: {job_name}")
            time.sleep(1)
        else:
            print(f"Created job file: {slurm_script_path}")

    if not args.submit:
        print("\nJobs were not submitted. Run 'sbatch *sh' or use the -s or --submit flag to auto-submit.")