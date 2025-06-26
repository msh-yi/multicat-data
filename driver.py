#!/usr/bin/env python

"""
driver.py: Module for simulating PD algorithms across multiple parameter sets.

This module implements a parallel processing framework for running one-shot pooling
experiments with various configurations. It handles landscape generation, pool
analysis, and result aggregation with support for interrupted runs.

The module processes multiple parameter combinations in parallel and evaluates
the effectiveness of different pooling strategies against given brute-force solutions.

Dependencies:
    - oneshot_pooler: Core pooling algorithm implementation
    - pool_cat_sim: Catalyst + simulation module
    - query_ljcr: Pool design query module
    - process_output: Module to parse output file

    March 19 2025
"""

import os
import sys
import glob
import warnings
import traceback
import argparse
from itertools import product
from multiprocess import Pool
import concurrent.futures

import yaml
import numpy as np
import pandas as pd

import oneshot_pooler
import pool_cat_sim
import query_ljcr
import process_output

# Suppress warnings
warnings.filterwarnings('ignore')


def setup_landscape(config, output_path):
    """
    Set up or load landscape based on configuration.

    Args:
        config (dict): Configuration dictionary from YAML file
        output_path (str): Path for output files, including prefix

    Returns:
        tuple: (ls_dicts, rate_file, coop_file, soln_ranks) containing:
            - ls_dicts: List of landscape dictionaries
            - rate_file: Path to rate file
            - coop_file: Path to cooperativity file
            - soln_ranks: Dictionary mapping combinations to their ranks
    """

    # Configure simulation parameters if specified
    if 'neg_strength' in config['simul']:
        pool_cat_sim.config(neg_strength=config['simul']['neg_strength'])

    if 'mode' in config['simul']:
        pool_cat_sim.config(coop_mech=config['simul']['mode'])

    # Either load existing landscape or generate a new one
    if 'landscape' in config['simul']:
        ls_path = config['simul']['landscape']
        files = glob.glob(f"{ls_path}*")
        rate_file = next((s for s in files if 'cat_rate' in s), None)
        coop_file = next((s for s in files if 'coop_mat' in s), None)
        
        ls_dicts = pool_cat_sim.set_landscape(rate_file, coop_file)
        print("Landscape read!")
        for d in ls_dicts:
            print(d)
            
        df_soln = pd.read_csv(f'{ls_path}_soln.csv')    
        df_soln['Combi'] = df_soln['Combi'].apply(eval)
        soln_ranks = {k: v for v, k in enumerate(df_soln['Combi'])}
        
        return ls_dicts, rate_file, coop_file, soln_ranks
    else:
        pool_cat_sim.gen_landscape(config_dict=config, out=output_path)
        print('Landscape(s) generated!')
        return None, None, None, None


def log_initial_info(output_path, config, ls_dicts):
    """Log initial information about the simulation run."""
    with open(output_path, 'w') as f:
        f.write('Beginning of run\n')
        f.write('landscape params:\n')
        for k, v in config['simul'].items():
            f.write(f'{k}: {v}\n')
        f.write('Landscape:\n')
        for d in ls_dicts:
            f.write(f'{d}')
        f.write('----------------------------\n\n')


def get_valid_configurations(config, output_path):
    """
    Generate valid parameter configurations for simulation.
    
    Args:
        config (dict): Configuration dictionary
        output_path (str): Path for output file
        
    Returns:
        list: List of valid configuration dictionaries
    """
    if 'oneshot' not in config:
        raise KeyError('The input yml file needs to contain a "oneshot" section.')
    
    hyper_config_dict = config['oneshot']
    
    # Generate all combinations of parameters
    keys = hyper_config_dict.keys()
    values = hyper_config_dict.values()
    config_dicts = [dict(zip(keys, combination)) for combination in product(*values)]
    
    N = config['simul']['N']
    valid_configs = []
    
    with open(output_path, 'a') as f:
        for config_dict in config_dicts:
            design = query_ljcr.query_ljcr(N, config_dict['pool_size'], config_dict['num_meet'])
            if not design:
                f.write(f'Warning: no covering exists for the following parameter set, so it will not be run\n')
                f.write(f'{config_dict}\n')
                continue
            if config_dict not in valid_configs:
                valid_configs.append(config_dict)
    
    return valid_configs


def filter_already_run_configs(valid_configs, interrupted_file):
    """Filter out configurations that have already been run."""
    if not interrupted_file:
        return valid_configs
        
    already_run_configs = process_output.process_out(interrupted_file, None, False)
    filtered_configs = [
        config for config in valid_configs 
        if not any(all(config[k] == run_config[k] for k in config) for run_config in already_run_configs)
    ]
    
    print(f'{len(filtered_configs)} combinations of parameters will be run')
    return filtered_configs

def run_oneshot_simulation(param_set, output_path, rate_file, soln_ranks,
                           pos_coop_sets, N, num_reps, seed,
                           verbose=False, pool_parallel=False):
    """
    Run one-shot simulation with the given parameter set.

    Returns
        (accuracy,
         coop_accuracy,
         mean_sensitivity,
         mean_rank,
         stdev_rank,
         mean_num_pools)
    """
    ranks            = []
    coop_indicators  = []   # True/False for “hit any cooperative pair”
    sensitivities    = []
    num_pools_list   = []

    try:
        with open(output_path, "a") as f:
            f.write("***********params************\n")
            for k, v in param_set.items():
                f.write(f"{k}: {v}\n")

            # ---------- single repetition ---------------------------------
            def run_single_rep(rep_idx):
                curr_top_set, num_pools = oneshot_pooler.run_oneshot(
                    N,
                    seed + rep_idx,
                    config_dict=param_set,
                    singles_file=rate_file,
                    logfile=output_path,
                    verbose=verbose,
                    pool_parallel=pool_parallel
                )

                # canonicalise once
                canon_pools = [tuple(pool) for pool in curr_top_set]

                # best rank
                best_rank = min(soln_ranks[p] for p in canon_pools)

                # cooperative pools retrieved
                coop_found = {p for p in canon_pools if p in pos_coop_sets}
                hit_any    = bool(coop_found)

                # sensitivity
                if pos_coop_sets:
                    sensitivity = len(coop_found) / len(pos_coop_sets)
                else:
                    sensitivity = 0.0

                return best_rank, hit_any, num_pools, sensitivity
            # ----------------------------------------------------------------

            with Pool(min(8, num_reps)) as p:
                rep_results = p.map(run_single_rep, range(num_reps))

            # unpack
            for rnk, hit, n_pools, sens in rep_results:
                ranks.append(rnk)
                coop_indicators.append(hit)
                num_pools_list.append(n_pools)
                sensitivities.append(sens)

            ranks_arr = np.array(ranks)

            accuracy        = np.mean(ranks_arr == 0)
            coop_accuracy   = np.mean(coop_indicators)
            mean_sens       = np.mean(sensitivities) if sensitivities else 0.0
            mean_rank       = np.mean(ranks_arr)      if ranks else float("inf")
            stdev_rank      = np.std(ranks_arr)       if ranks else float("inf")
            mean_num_pools  = np.mean(num_pools_list) if num_pools_list else 0.0

            # log
            f.write("Best ranks per rep:\n")
            f.write(f"{ranks}\n")
            f.write(
                f"accuracy: {accuracy}; "
                f"coop_accuracy: {coop_accuracy}; "
                f"mean sensitivity: {mean_sens}; "
                f"avg rank: {mean_rank}; "
                f"stdev rank: {stdev_rank}; "
                f"total pools: {mean_num_pools}\n\n"
            )

            return (accuracy,
                    coop_accuracy,
                    mean_sens,
                    mean_rank,
                    stdev_rank,
                    mean_num_pools)

    except Exception:
        traceback.print_exc()
        return (0.0, 0.0, 0.0, float("inf"), float("inf"), float("inf"))

def run_oneshot_simulation_old(param_set, output_path, rate_file, soln_ranks, pos_coop_sets, N, num_reps, seed, verbose=False, pool_parallel=False):
    """
    Run one-shot simulation with the given parameter set.
    
    Args:
        param_set (dict): Dictionary of parameters
        output_path (str): Path for output file
        rate_file (str): Path to rate file
        soln_ranks (dict): Dictionary of solution ranks
        pos_coop_sets (list): List of positive cooperative sets
        N (int): Size of the system
        num_reps (int): Number of repetitions
        seed (int): Random seed
        verbose (bool): Whether to output verbose information
        pool_parallel (bool): Whether to parallelize pool processing
        
    Returns:
        tuple: (accuracy, coop_accuracy, mean_rank, stdev_rank, mean_num_pools)
    """
    ranks = []
    identified_coop = []
    num_pools_list = []
    
    try:
        with open(output_path, 'a') as f:
            f.write('***********params************\n')
            for k, v in param_set.items():
                f.write(f'{k}: {v}\n')

            def run_single_rep(number):
                """Run a single repetition of one-shot pooling."""
                curr_top_set, num_pools = oneshot_pooler.run_oneshot(
                    N, seed + number, config_dict=param_set, 
                    singles_file=rate_file, logfile=output_path, 
                    verbose=verbose, pool_parallel=pool_parallel
                )
                
                # Calculate ranks and check if top set is cooperative
                curr_top_set_ranks = []
                for pool in curr_top_set:
                    pool = tuple(list(pool))
                    curr_top_set_ranks.append(soln_ranks[pool])

                curr_top_set_coop = [int(tuple(list(pool)) in pos_coop_sets) for pool in curr_top_set]

                return min(curr_top_set_ranks), any(curr_top_set_coop), num_pools

            # Run repetitions in parallel
            with Pool(min(8, num_reps)) as p:
                one_param_set_results = p.map(run_single_rep, range(num_reps))

            # Collect results
            for curr_top_rank, curr_top_coop, num_pools in one_param_set_results:
                ranks.append(curr_top_rank)
                identified_coop.append(curr_top_coop)
                num_pools_list.append(num_pools)
            
            # Calculate statistics
            f.write('Best ranks:\n')
            f.write(f'{ranks}\n')
            ranks = np.array(ranks)
            accuracy = sum(1 for rank in ranks if rank == 0) / len(ranks)
            coop_accuracy = sum(identified_coop) / len(ranks)
            mean_rank = np.mean(ranks)
            stdev_rank = np.std(ranks)
            mean_num_pools = np.mean(num_pools_list)

            f.write(f'accuracy: {accuracy}; coop_accuracy: {coop_accuracy}; '
                   f'avg rank: {mean_rank}; stdev rank: {stdev_rank}; '
                   f'total pools: {mean_num_pools}\n\n')
                   
            return accuracy, coop_accuracy, mean_rank, stdev_rank, mean_num_pools
    except Exception:
        traceback.print_exc()
        # Return default values in case of error
        return 0.0, 0.0, float('inf'), float('inf'), float('inf')


def simulate_PD(config=None, out=None, verbose=False, interrupted=None, pool_parallel=False):
    """
    Execute PD simulations with given parameter sets.

    Args:
        config (dict): Configuration dictionary, usually from YAML file
        out (str): Output file path prefix
        verbose (bool): Enable verbose output. Defaults to False.
        interrupted (str): Path to previous output file for resumed runs
        pool_parallel (bool): Enable parallel pool processing. Defaults to False.

    Raises:
        KeyError: If required configuration sections are missing
    """
    # Set up landscape
    ls_dicts, rate_file, coop_file, soln_ranks = setup_landscape(config, out)
    
    # Log initial information
    log_initial_info(out, config, ls_dicts)
    
    # Get positive cooperative sets from the landscape
    global pos_coop_sets
    pos_coop_sets = list(ls_dicts[1].keys())
    
    # Get valid configurations
    valid_configs = get_valid_configurations(config, out)
    
    # Filter out already run configurations if resuming
    if interrupted:
        valid_configs = filter_already_run_configs(valid_configs, interrupted)
    
    # Get simulation parameters
    N = config['simul']['N']
    num_reps = config['simul']['num_reps']
    seed = config['rng']['seed']
    
    # Run simulations in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(12, len(valid_configs))) as executor:
        # Map each parameter set to a simulation function
        futures = [
            executor.submit(
                run_oneshot_simulation,
                param_set, out, rate_file, soln_ranks, pos_coop_sets,
                N, num_reps, seed, verbose, pool_parallel
            )
            for param_set in valid_configs
        ]
        
        # Collect results
        ranks_stats = [future.result() for future in concurrent.futures.as_completed(futures)]
    
    # ---------- pick best parameter set ---------------------------------
    if ranks_stats:
        best_idx = min(                                          # same idea
            range(len(ranks_stats)),
            key=lambda idx: (
                -ranks_stats[idx][0],        # maximise accuracy
                -ranks_stats[idx][2],        # maximise mean_sensitivity 
                ranks_stats[idx][5],         # minimise num_pools
                -ranks_stats[idx][1]         # maximise coop_accuracy
            )
        )

        # Log best result
        with open(out, 'a') as f:
            f.write('\n--------------------------------------------\n')
            f.write('Final analysis\n')
            best_result = ranks_stats[best_idx]
            f.write(
                f'Highest accuracy: {best_result[0]}; '
                f'mean sensitivity: {best_result[2]}; '
                f'total pools: {best_result[5]}; Params:\n'
            )
            f.write(f'{valid_configs[best_idx]}\n')

        # ----------------- build DataFrame ------------------------------
        config_df = pd.DataFrame(valid_configs)

        stats_df = pd.DataFrame(
            ranks_stats,
            columns=[
                'accuracy',
                'coop_accuracy',
                'mean_sensitivity',
                'mean_rank',
                'stdev_rank',
                'num_pools'
            ]
        )

        results_df = pd.concat([config_df, stats_df], axis=1)

        # Sort: high accuracy, high sensitivity, few pools, high coop_acc
        results_df = results_df.sort_values(
            by=['accuracy', 'mean_sensitivity', 'num_pools', 'coop_accuracy'],
            ascending=[False,         False,        True,          False]
        )

        # Write to CSV
        results_df.to_csv(f'{out}.csv', index=False)


def main():
    parser = argparse.ArgumentParser(description='Run PD simulations with various configurations.')
    parser.add_argument("-c", "--config", help="path of yaml config file", required=True)
    parser.add_argument("-v", "--verbose", help="increase verbosity", action='store_true')
    parser.add_argument("-g", "--genls", help="if activated, only generate landscape but not run", action='store_true')
    parser.add_argument("-o", "--outpath", help="path to output file")
    parser.add_argument("-i", "--interrupted", help="previous output file", default=None)
    parser.add_argument("-m", "--parallelpools", help="parallelize pool processing", action='store_true')

    args = parser.parse_args()

    # Load configuration
    try:
        with open(args.config, 'r') as stream:
            config = yaml.safe_load(stream)
    except yaml.YAMLError as e:
        print(e)
        sys.exit(1)
    except FileNotFoundError:
        print(f"Config file {args.config} not found")
        sys.exit(1)

    config_abs_path = os.path.abspath(args.config)
    curr_dir = os.path.dirname(config_abs_path)
    project_name = os.path.splitext(os.path.basename(args.config))[0]
    if args.genls:
        #ls_dir = f'{curr_dir}/ls'
        ls_dir = curr_dir 
        #os.makedirs(ls_dir, exist_ok=True)
        ls_output_path = f'{ls_dir}/{project_name}_ls'
        setup_landscape(config, ls_output_path)
    else:
        # Run full simulation
        out_dir = os.path.dirname(args.outpath)
        os.makedirs(out_dir, exist_ok=True)
        simulate_PD(
            config=config, 
            out=args.outpath,
            verbose=args.verbose, 
            interrupted=args.interrupted, 
            pool_parallel=args.parallelpools
        )


if __name__ == "__main__":
    main()