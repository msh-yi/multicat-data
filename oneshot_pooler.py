#!/usr/bin/env sage -python

"""
oneshot_pooler.py: Implementation of one-shot pooling algorithms for catalyst discovery.

This module provides functionality for evaluating pool performance, scoring cooperativity,
and deconvoluting pools to find optimal catalyst combinations. It can operate in different
modes: simulation, recommendation, or processing of experimental data.
"""

import os
import sys
import math
import ast
import yaml
from datetime import datetime
from itertools import combinations
from collections import defaultdict
import multiprocess
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from deap import base
from deap import creator

import pool_cat_sim
import query_ljcr


# Global RNG variable
rng = None


def read_pools(pool_class, file, rxn_time=None, use_k=False):
    """
    Read pool data from a CSV file containing pools and their outputs.
    
    Args:
        pool_class: Pool class from DEAP creator
        file (str): Path to CSV file
        rxn_time (float, optional): Reaction time for k_eff calculation
        use_k (bool): If True, converts output to k_eff
        
    Returns:
        list: List of Pool objects with fitness values assigned
        
    Raises:
        ValueError: If output values are missing
    """
    df = pd.read_csv(file)
    df['pool'] = df['pool'].apply(ast.literal_eval)

    # Convert to 0-indexed pools
    pools_list = df['pool'].tolist()
    pools_list_0 = [[idx - 1 for idx in pool] for pool in pools_list]
    pop = [pool_class(pool) for pool in pools_list_0]
    
    # Process outputs
    outputs = df['output'].tolist()
    outputs = [float(y) for y in outputs]
    if max(outputs) >= 1:  # yields in percent, not fraction
        outputs = [y/100 for y in outputs]

    # Calculate fitnesses
    if use_k and rxn_time:
        fitnesses = [-np.log(1-x) / rxn_time if x < 1 else float('inf') for x in outputs]
    else:
        fitnesses = outputs
    
    if any(np.isnan(fitnesses)):  # hasn't been filled in
        raise ValueError("The file does not have the expected output values for this generation!")

    # Assign fitness values to pool objects
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = (fit, 0)

    return pop


def read_singles(mode, singles_file, rxn_time=None, use_k=False):
    """
    Read data for single catalysts from a file.
    
    Args:
        mode (str): 'sim' or 'proc' mode
        singles_file (str): Path to file with singles data
        rxn_time (float, optional): Reaction time for k_eff calculation
        use_k (bool): If True, converts yield to k_eff
        
    Returns:
        list: List of single catalyst outputs or k values
        
    Raises:
        ValueError: If reaction time is missing when needed
    """
    singles_df = pd.read_csv(singles_file, header=None)
    
    # Remove header row if present
    if 'output' in singles_df.iloc[0].values:
        singles_df.drop(index=singles_df.index[0], axis=0, inplace=True)
    
    if mode == 'sim':  # singles k's are in file
        singles_outputs = singles_df.iloc[:, 1].tolist()
    else:  # need to calculate k's from yields
        if not rxn_time:
            raise ValueError('No reaction time specified for k_cat calculation!')
            
        singles_yields = singles_df.iloc[:, 1].tolist()
        singles_yields = [float(y) for y in singles_yields]
        if max(singles_yields) >= 1:  # yields in percent, not fraction
            singles_yields = [y/100 for y in singles_yields]
            
        if use_k and rxn_time:
            singles_outputs = [-np.log(1-x)/rxn_time if x < 1 else float('inf') for x in singles_yields]
        else:
            singles_outputs = singles_yields
    
    return singles_outputs


def eval_yield(pool, noise=True):
    """
    Evaluate the effective rate constant (k_eff) of a pool using simulation.
    
    Args:
        pool (list): List of catalyst indices or binary representation
        noise (bool): If True, add random noise to non-zero values
        
    Returns:
        float: Effective rate constant (k_eff) of the pool
    """
    rxn = pool_cat_sim.Reaction.create(pool)
    k = rxn.k_eff
    
    # Add noise to non-trivial values
    if noise and k > 0.1:
        # We don't want a k of zero to become nonzero; 
        # esp if the non-coop baseline is zero, a nonzero k might give 
        # an astronomical Q because of how we implemented score_coop
        k += np.random.normal(0, 1)  # mean error about 1, std dev 0.2
        k = max(k, 0)  # at least 0
        
    return k


def score_coop(score_mode, pool, singles_ks):
    """
    Calculate cooperativity score for a pool based on individual catalyst performances.
    
    Args:
        score_mode (str): 'sum', 'avg', or 'max' - how to compare against singles
        pool (list): List representing a pool of catalysts
        singles_ks (list): List of k values for individual catalysts
        
    Returns:
        float: Cooperativity score
    """
    k_comp = 1E-10  # to avoid div by zero, since the singles k_s might be zero
    singles_ks_this_pool = [singles_ks[cat] for cat in pool]
    
    # Calculate baseline for comparison
    if score_mode == 'max':  # baseline is maximum of singles
        k_comp += max(singles_ks_this_pool)
    else:  # baseline is sum/average of singles
        k_comp += sum(singles_ks_this_pool)
    
    # Calculate cooperativity score
    if score_mode == 'avg':  # scale up for comparison with k_comp
        score = pool.fitness.values[0] * len(pool)/k_comp - 1
    elif score_mode in ['sum', 'max']:
        score = pool.fitness.values[0]/k_comp - 1
        
    return score


def init_covering(pool_class, num_cats, pool_size, num_meet, num_redun):
    """
    Initialize a population of pools using a covering design.
    
    Args:
        pool_class: Pool class from DEAP
        num_cats (int): Total number of catalysts
        pool_size (int): Size of each pool
        num_meet (int): How many times each pair of catalysts should meet
        num_redun (int): How many times to repeat the covering with different permutations
        
    Returns:
        list: List of Pool objects representing the initial population
    """
    # Get base design from LJCR query
    design = query_ljcr.query_ljcr(num_cats, pool_size, num_meet)
    design = [[(cat_idx - 1) for cat_idx in pool] for pool in design]
    
    pools = []
    
    # Create redundant copies with permuted catalyst indices
    for _ in range(num_redun):
        labels = np.arange(num_cats)
        shuffled_labels = rng.permutation(labels)
        label_mapping = dict(zip(labels, shuffled_labels))

        # Apply mapping to generate new permuted pools
        new_groups = [[label_mapping[label] for label in pool] for pool in design]
        pools.extend(new_groups)
    
    # Sort pools and convert to Pool objects
    pools = sorted([sorted(pool) for pool in pools])
    pools = [pool_class(pool) for pool in pools]
    
    return pools


def tiebreak_init(pool_class, pop):
    """
    Convert a list of integer lists to Pool objects.
    
    Args:
        pool_class: Pool class from DEAP
        pop (list): List of lists of integers
        
    Returns:
        list: List of Pool objects
    """
    return [pool_class(pool) for pool in pop]


def deconvolute(frac_top, final_size, target_metric, num_top, pop, verbose):
    """
    Deconvolute pools to find optimal subsets of catalysts.
    
    Args:
        frac_top (float): Fraction of top pools to consider
        final_size (int): Target size for deconvoluted pools
        target_metric (str): 'coop' or 'output' - metric to optimize
        num_top (int): Number of top results to return
        pop (list): Population of Pool objects
        verbose (bool): Whether to print detailed output
        
    Returns:
        tuple: (top_pools, sorted_intersections) where:
            top_pools: List of top-performing pools of size final_size
            sorted_intersections: Complete sorted list of pool-score pairs
    """
    # Sort pools by the target metric
    if target_metric == 'coop':
        sorted_pools = sorted(pop, key=lambda x: x.fitness.values[1], reverse=True)
    elif target_metric == 'output':
        sorted_pools = sorted(pop, key=lambda x: x.fitness.values[0], reverse=True)

    # If pools are already at final size, no deconvolution needed
    if len(sorted_pools[0]) == final_size:
        return sorted_pools[:num_top]

    # Take top fraction of pools
    index = math.ceil(frac_top * len(pop))
    top_pools = sorted_pools[:index]

    # Calculate average score for each final_size combination
    intersection_dict = defaultdict(list)
    for pool in top_pools:
        for comb in combinations(pool, final_size):
            intersection = tuple(sorted(comb))
            if target_metric == 'coop':
                intersection_dict[intersection].append(pool.fitness.values[1])
            elif target_metric == 'output':
                intersection_dict[intersection].append(pool.fitness.values[0])

    # Calculate average score for each intersection
    for key in intersection_dict:
        intersection_dict[key] = sum(intersection_dict[key]) / len(intersection_dict[key])

    # Sort intersections by score
    sorted_intersections = sorted(intersection_dict.items(), key=lambda x: x[1], reverse=True)
    
    # Print top results if verbose
    if verbose:
        for i, (pair, score) in enumerate(sorted_intersections[:10]):
            print(f"{i+1}. {pair}: {score}")
    
    # Include ties that are close to the last qualifying entry
    last_qual_entry = sorted_intersections[num_top-1][1]
    tie_threshold = 0.1
    num_ties = sum(1 for entry in sorted_intersections[num_top:] 
                   if (last_qual_entry - entry[1]) <= tie_threshold)
    num_top += num_ties
    
    # Get the top intersections
    intersections = [list(item[0]) for item in sorted_intersections]
    return intersections[:num_top], sorted_intersections


def pre_config_oneshot(N, pool_size, num_meet, num_redun, frac_top, final_size, 
                       num_top, target_metric, score_mode, mode, **kwargs):
    """
    Configure DEAP toolbox for oneshot pooling.
    
    Args:
        N (int): Total number of catalysts
        pool_size (int): Size of each pool
        num_meet (int): How many times each pair should meet
        num_redun (int): How many redundant coverings
        frac_top (float): Fraction of top pools to consider
        final_size (int): Size of final pools
        num_top (int): Number of top results to return
        target_metric (str): 'coop' or 'output' - metric to optimize
        score_mode (str): 'sum', 'avg', or 'max' - how to compare against singles
        mode (str): 'sim', 'rec', or 'proc' - operating mode
        **kwargs: Additional parameters
        
    Returns:
        base.Toolbox: Configured DEAP toolbox
    """
    # Create fitness and individual classes
    creator.create("RateMax", base.Fitness, weights=(1.0, 1.0))
    # Fitness values are k_tot and cooperativity
    creator.create("Pool", list, fitness=creator.RateMax)

    # Create toolbox and register functions
    toolbox = base.Toolbox()

    if mode in ['sim', 'rec']:
        toolbox.register("start", init_covering, creator.Pool, N, pool_size, num_meet, num_redun)
        toolbox.register("evaluate", eval_yield)
    else:  # pool recommender mode
        toolbox.register("read", read_pools, creator.Pool)  # includes eval
        
    toolbox.register("read_singles", read_singles, mode)
    toolbox.register("score_coop", score_coop, score_mode)
    toolbox.register("deconvolute", deconvolute, frac_top, final_size, target_metric, num_top)
    toolbox.register("tiebreak_init", tiebreak_init, creator.Pool)
    
    return toolbox


def score_pools(pop, mode, toolbox, singles_ks, pool_parallel=False):
    """
    Score a population of pools for both output and cooperativity.
    
    Args:
        pop (list): Population of Pool objects
        mode (str): 'sim', 'rec', or 'proc' mode
        toolbox: DEAP toolbox with registered functions
        singles_ks (list): k values for individual catalysts
        pool_parallel (bool): Whether to use parallel processing
        
    Returns:
        tuple: (pop, fitnesses, coop_scores) - scored population and results
    """
    # Score pools for output
    if mode == 'sim':
        if pool_parallel:  # multiprocess
            num_procs = max(1, int(multiprocess.cpu_count()/8))
            with ThreadPoolExecutor(max_workers=num_procs) as executor:
                print(f'Using {num_procs} processes for threading!')
                fitnesses = list(executor.map(toolbox.evaluate, pop))
        else:  # do not multiprocess
            fitnesses = list(map(toolbox.evaluate, pop))
            
        # Assign output fitness values
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = (fit, 0)
    else:  # pool items already have fitnesses
        fitnesses = [pool.fitness.values[0] for pool in pop]

    # Score pools for cooperativity
    coop_scores = [toolbox.score_coop(pool, singles_ks) for pool in pop]
    
    # Update fitness values with cooperativity scores
    for ind, coop_score in zip(pop, coop_scores):
        ind.fitness.values = (ind.fitness.values[0], coop_score)
    
    return pop, fitnesses, coop_scores


def run_oneshot(N, seed, config_file=None, config_dict=None, singles_file=None, 
                logfile=None, verbose=False, mode='sim', infile=None, pool_parallel=False):
    """
    Run oneshot pooling to find optimal catalyst combinations.
    
    Args:
        N (int): Total number of catalysts
        seed (int): Random seed
        config_file (str, optional): Path to YAML config file
        config_dict (dict, optional): Configuration dictionary
        singles_file (str, optional): Path to file with singles data
        logfile (str, optional): Path to log file
        verbose (bool): Whether to print detailed output
        mode (str): 'sim', 'rec', or 'proc' - operating mode
        infile (str, optional): Input file for proc mode
        pool_parallel (bool): Whether to use parallel processing
        
    Returns:
        tuple: (top_set, num_pools) or (top_set, sorted_pop_with_scores)
        
    Raises:
        ValueError: If required parameters are missing
    """
    global rng  
    rng = np.random.default_rng(seed=seed)
    
    # Load configuration
    if config_file:
        with open(config_file, 'r') as stream:
            try:
                config = yaml.safe_load(stream)
            except yaml.YAMLError as e:
                print(e)
                sys.exit(1)
        config = config['oneshot']
        # For rec or proc mode, take first value if parameters are lists
        if mode in ['rec', 'proc']:
            for k, v in config.items():
                if isinstance(v, list): 
                    config[k] = v[0]
    else:
        config = config_dict
    
    # Configure toolbox
    toolbox = pre_config_oneshot(N, config['pool_size'], config['num_meet'], config['num_redun'], 
                                config['frac_top'], config['final_size'], config['num_top'], 
                                config['target_metric'], config['score_mode'], mode=mode)
    
    # Initialize population based on mode
    if mode == 'sim':
        pop = toolbox.start()  # returns the covering
        singles_ks = toolbox.read_singles(singles_file)
    elif mode == 'proc':  # real mode
        if not config.get('rxn_time'):
            raise ValueError('No reaction time given!')
        config['rxn_time'] = float(config['rxn_time'])
        pop = toolbox.read(infile, config['rxn_time'])  # should assign fitnesses
        singles_ks = toolbox.read_singles(singles_file, config['rxn_time'])

        # Process tiebreak round results if applicable
        if len(pop[0]) == config['final_size']:
            pop, _, _ = score_pools(pop, mode, toolbox, singles_ks)
            
            # Choose sorting criterion
            if config['target_metric'] == 'coop':
                sorted_pools = sorted(pop, key=lambda x: x.fitness.values[1], reverse=True)
                sorted_pop_with_scores = [(ind, ind.fitness.values[1]) for ind in sorted_pools]
            else:  # output
                sorted_pools = sorted(pop, key=lambda x: x.fitness.values[0], reverse=True)
                sorted_pop_with_scores = [(ind, ind.fitness.values[0]) for ind in sorted_pools]
                
            #top_set = [sorted_pools[0]]
            #top_set = [tuple(pool[:]) for pool in top_set]
            top_set = [tuple(pool[:]) for pool in sorted_pools]
            return top_set, sorted_pop_with_scores
    elif mode == 'rec':
        pop = toolbox.start()  # returns the covering
        return pop, None
    
    # Open log file if provided
    log_file = open(logfile, 'a') if logfile else None
    
    try:
        # Log initial information in sim mode
        if log_file and mode == 'sim':
            log_file.write(f'{datetime.now()}\n')
            if verbose:
                log_file.write("Pools:\n")
                log_file.write(f"{pop}\n")
                log_file.write("Singles:\n")
                log_file.write(f"{singles_ks}\n")
        
        # Score pools
        pop, fitnesses, coop_scores = score_pools(pop, mode, toolbox, singles_ks, pool_parallel)
        num_pools = len(pop)
        
        # Log scoring results in sim mode
        if log_file and verbose and mode == 'sim':
            log_file.write("Fitnesses are:\n")
            log_file.write(f"{fitnesses}\n")
            log_file.write("Coop scores are:\n")
            log_file.write(f"{coop_scores}\n")
        
        # Deconvolute to find top combinations
        top_set, sorted_pop_avgscores = toolbox.deconvolute(pop, verbose)

        # Run tiebreak round if requested
        if config.get('tiebreak_round', False): # returns true if tiebreak_round exists in yaml, else returns False
            pop = toolbox.tiebreak_init(top_set)
            num_pools += len(top_set)

            if mode == 'sim':
                pop, fitnesses, coop_scores = score_pools(pop, mode, toolbox, singles_ks, pool_parallel)
            elif mode == 'proc':  # proc can also take tiebreak rounds
                return pop, sorted_pop_avgscores

            # Log tiebreak results
            if log_file and verbose and mode == 'sim':
                log_file.write("----------Tiebreak round----------\n")
                log_file.write("Pools:\n")
                log_file.write(f"{pop}\n")
                log_file.write("Fitnesses are:\n")
                log_file.write(f"{fitnesses}\n")
                log_file.write("Coop scores are:\n")
                log_file.write(f"{coop_scores}\n")

            # Choose final top set
            if config['target_metric'] == 'coop':
                sorted_pools = sorted(pop, key=lambda x: x.fitness.values[1], reverse=True)
            else:  # output
                sorted_pools = sorted(pop, key=lambda x: x.fitness.values[0], reverse=True)
                
            #top_set = [sorted_pools[0]]
            top_set = sorted_pools

        top_set = [tuple(pool[:]) for pool in top_set]
        return top_set, num_pools
    
    finally:
        if log_file:
            log_file.close()


def real_oneshot(online=False, outfile=None, **kwargs):
    """
    Public interface function for recommending pools or processing real data.
    
    Args:
        online (bool): Whether running in online mode
        outfile (str, optional): Path for output file
        **kwargs: Arguments to pass to run_oneshot
        
    Returns:
        pandas.DataFrame: DataFrame with pools and scores
        
    Notes:
        In rec mode, returns pools to run.
        In proc mode, returns pools and their scores.
    """
    # Set default values for config dict
    if 'config_dict' in kwargs:
        if 'frac_top' not in kwargs['config_dict']:
            kwargs['config_dict']['frac_top'] = 1.0
        if 'tiebreak_round' not in kwargs['config_dict']:
            kwargs['config_dict']['tiebreak_round'] = True
        if 'score_mode' not in kwargs['config_dict']: # for rec mode we don't ask user for score mode
            kwargs['config_dict']['score_mode'] = 'avg'
            
    # Run oneshot algorithm
    pop, sorted_pop_w_scores = run_oneshot(**kwargs)
    
    if kwargs['mode'] == 'rec':
        # we get only pop back
        pop_out = [[cat + 1 for cat in pool] for pool in pop] 
        df = pd.DataFrame({
            'pool': pop_out,
            'output': None  # Column for user to fill in
        })
        

    elif kwargs['mode'] == 'proc':
    # Convert 0-indexed to 1-indexed for user-facing output
        pop_out = [[cat + 1 for cat in tup[0]] for tup in sorted_pop_w_scores]
        scores = [tup[1] for tup in sorted_pop_w_scores]

        # Create output DataFrame
        df = pd.DataFrame({
            'pool': pop_out,
            'score': scores,
            'output': None  # Column for user to fill in
        })
        
    # Save to file if not in online mode
    if not online and outfile:
        df.to_csv(outfile, index=False)
    
    return df


if __name__ == '__main__':
    # Example usage
    config_file = '/home/msh_yi/research/multicat/validation/expt_inputs/par_exp_73-76_agg_y/72_config_merck_deCOSMC.yaml'

    with open(config_file, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as e:
            print(e)
            sys.exit(1)
    
    # Process configuration
    config = config['oneshot']
    for k, v in config.items():
        if isinstance(v, list): 
            config[k] = v[0]
    
    # Example: process tiebreak round
    df_out = real_oneshot(
        online=False, 
        N=72, 
        seed=1234, 
        config_dict=config, 
        singles_file='/home/msh_yi/research/multicat/validation/expt_inputs/par_exp_73-76_agg_y/singles_73_75_biaryl_y.csv', 
        logfile='./log_waste.log', 
        verbose=False, 
        mode='proc', 
        infile='/home/msh_yi/research/multicat/validation/expt_inputs/par_exp_73-76_agg_y/par_exp_77_biaryl_y.csv'
    )

    # Save output
    output_path = '/home/msh_yi/research/multicat/validation/expt_inputs/par_exp_73-76_agg_y/par_exp_77_tieb_results_biaryl.csv'
    df_out.to_csv(output_path, index=False)
    print(df_out)