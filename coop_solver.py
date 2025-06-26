#!/usr/bin/env python

"""
coop_solver.py: Brute-force solver for catalyst pools

After generating hypothetical landscapes (cat_rate and coop_mat), use this module to compute complete landscapes, which are necessary for evaluating outcomes of simulated PDs. It takes a landscape, enumerates all sets up to maximum_size exhaustively, and calculates rate constants for those pairs accounting for cooperativity. Parallelization is used where possible.

Dependencies:
    - pool_cat_sim: Custom simulation module for catalytic reactions
    - multiprocess: For parallel processing of catalyst combinations

Author: Marcus Sak
Date: 3/18/25
"""

from functools import partial
import argparse
import ast
import csv
import glob
import itertools
import multiprocess
import os
import os
import pool_cat_sim
import re
import sys

#@profile
def eval_output(cat_list, rate_file, coop_file, mode, neg_strength):
    """
    Evaluate the output of a catalytic reaction for a given combination of catalysts.

    Args:
        cat_list (list): List of catalyst indices to evaluate
        rate_file (str): Path to the catalyst rate file
        coop_file (str): Path to the cooperativity matrix file
        mode (str): Mode of cooperativity, usually "dimer"
        neg_strength (int): Binding strength of negative cooperative pairs (default 100)

    Returns:
        float: Effective reaction rate for the catalyst combination
    """
    pool_cat_sim.config(coop_mech=mode, neg_strength=neg_strength)
    pool_cat_sim.set_landscape(rate_file, coop_file)
    rxn = pool_cat_sim.Reaction.create(cat_list)
    return rxn.output

def solve_coop_mp(N, rate_file, coop_file, max_pool_size=5, outfile=None, neg_strength=100, mode='dimer'):
    """
    Solve cooperative catalysis problems using multiprocessing.

    Args:
        N (int): Number of catalysts
        rate_file (str): Path to catalyst rate file
        coop_file (str): Path to cooperativity matrix file
        max_pool_size (int, optional): Maximum size of catalyst pools. Defaults to 5.
        outfile (str, optional): Path for output file. Defaults to None.
        neg_strength (int, optional): Negative cooperation strength. Defaults to 100.
        mode (str, optional): Cooperation mechanism mode. Defaults to 'dimer'.

    Returns:
        list: Sorted list of tuples (catalyst_combination, output_rate)
    """
    combinations = [list(itertools.combinations(range(N), i)) for i in range(1, max_pool_size + 1)]
    combi = [item for sublist in combinations for item in sublist]

    total = len(combi)
    print(f'Total {total} pools expected')

    # Prepare the eval_output function with fixed parameters
    eval_output_partial = partial(eval_output, rate_file=rate_file, coop_file=coop_file, mode=mode, neg_strength=neg_strength)

    # Use multiprocessing Pool
    with multiprocess.Pool(processes=int(multiprocess.cpu_count()/4)) as pool:
        outputs = pool.map(eval_output_partial, combi)

    outputs = list(zip(combi, outputs))
    outputs.sort(key=lambda x: x[1], reverse=True)

    # Write results to file if outfile is specified
    if outfile:
        with open(f'{outfile}_soln.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Combi", "k_eff"])
            for row in outputs:
                writer.writerow(row)

    return outputs

def csv_to_kv(file, coop=False):
    """
    Convert CSV file to key-value dictionaries.

    Args:
        file (str): Path to CSV file
        coop (bool, optional): If file contains cooperativity  (i.e. "*_coop_mat.csv), True. Defaults to False.

    Returns:
        Union[dict, tuple[dict, dict]]: Single dictionary for rates or tuple of dictionaries (positive, negative) for cooperative data
    """
    csv_dict = {}
    pos_dict = {}
    neg_dict = {}
    with open(file, 'r') as f:
            
        # Create a CSV reader
        reader = csv.reader(f)

        if not coop:
            # Loop over the rows in the CSV file
            for row in reader:
                # Add the key-value pair to the dictionary
                key, value = row[0], ast.literal_eval(row[1])
                csv_dict[int(key)] = value
        else:
            for row in reader:
                key, value = ast.literal_eval(row[0]), ast.literal_eval(row[1])
                if value >= 1:
                    pos_dict[key] = float(value)
                else:
                    neg_dict[key] = float(value)
    if not coop:
        return csv_dict
    else:
        return pos_dict, neg_dict

def natural_sort_key(s):
    return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', s)]

def process_files(rate_file, coop_file, args, basename):
    """
    Process a set of rate and cooperativity files.

    Args:
        rate_file (str): Path to rate file
        coop_file (str): Path to cooperativity file
        args (argparse.Namespace): Command line arguments
        basename (str): Base name for output files

    Raises:
        FileNotFoundError: If input files are not found
    """

    if not os.path.isfile(rate_file):
        raise FileNotFoundError(f"Rate file '{rate_file}' not found.")

    if not os.path.isfile(coop_file):
        raise FileNotFoundError(f"Cooperativity matrix file '{coop_file}' not found.")

    cat_rate_dict = csv_to_kv(rate_file)
    N = len(cat_rate_dict)

    solve_coop_mp(N, rate_file, coop_file, max_pool_size=args.poolsize, outfile=basename, mode=args.coopmode, neg_strength=args.negstrength)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="path to landscapes directory containing *cat_rate.csv files, or path to a specific cat_rate.csv file when not in batch mode")
    parser.add_argument("-k", "--poolsize", type=int, help="maximum pool size to solve for", default=3)
    parser.add_argument("-m", "--coopmode", help="mode of cooperativity", default='dimer')
    parser.add_argument("-s", "--negstrength", type=int, help="strength of negative cooperative dimers", default=100)
    parser.add_argument("-b", "--batch", help="solve a batch of landscapes", action='store_true')

    args = parser.parse_args()
    # Convert input to absolute path to handle both relative and absolute paths
    input_path = os.path.abspath(args.input)

    if args.batch:
        # In batch mode, find all cat_rate files in the directory
        if not os.path.isdir(input_path):
            print(f"Error: {input_path} is not a directory. In batch mode, input must be a directory path.")
            sys.exit(1)
        
        all_rate_files = glob.glob(f"{input_path}/*cat_rate.csv")
        
        if not all_rate_files:
            print(f"No cat_rate.csv files found in {args.input}")
            sys.exit(1)
            
        # Verify each cat_rate file has a corresponding coop_mat file
        valid_prefixes = []
        for rate_file in all_rate_files:
            prefix = rate_file[:-13]  # Remove "_cat_rate.csv"
            coop_file = f"{prefix}_coop_mat.csv"
            
            if os.path.exists(coop_file):
                valid_prefixes.append(prefix)
            else:
                print(f"Warning: Found {rate_file} but missing {coop_file} - skipping")
        
        if not valid_prefixes:
            print("No valid file pairs found with both cat_rate.csv and coop_mat.csv")
            sys.exit(1)
            
        # Sort prefixes naturally
        sorted_prefixes = sorted(valid_prefixes, key=natural_sort_key)

    else:
        # Not in batch mode - user provided a specific cat_rate file
        if not args.input.endswith("_cat_rate.csv"):
            print("In non-batch mode, input must be a path to a specific *_cat_rate.csv file")
            sys.exit(1)
            
        rate_file = args.input
        prefix = rate_file[:-13]  # Remove "_cat_rate.csv"
        coop_file = f"{prefix}_coop_mat.csv"
        
        if not os.path.exists(rate_file):
            print(f"Specified cat_rate file not found: {rate_file}")
            sys.exit(1)
            
        if not os.path.exists(coop_file):
            print(f"Corresponding coop_mat file not found: {coop_file}")
            sys.exit(1)
            
        sorted_prefixes = [prefix]

    # Process each valid prefix
    for prefix in sorted_prefixes:
        rate_file = f"{prefix}_cat_rate.csv"
        coop_file = f"{prefix}_coop_mat.csv"
        basename = prefix
        
        # Extract the index if in batch mode
        index = ""
        if args.batch:
            match = re.search(r"_(\d+)$", prefix)
            if match:
                index = match.group(1)
        
        try:
            print(f"Processing files{' with index ' + index if index else ''}")
            process_files(rate_file, coop_file, args, basename)
        except Exception as e:
            print(f"Error processing {basename}: {e}")
            if not args.batch:
                raise  # Re-raise the exception if not in batch mode
        
        if not args.batch:
            break  # Not needed with current logic but kept for clarity