#!/usr/bin/env python

"""
process_output.py: Parse and process output files from simulation runs.

This module extracts parameter information and performance metrics from
simulation output files and optionally saves the results to a CSV file.
"""

import re
import sys
import ast
import pandas as pd
from typing import List, Dict, Union, Optional, Any


def process_out(input_file: str, output_file: Optional[str] = None, out_csv: bool = True) -> Union[List[Dict[str, Any]], None]:
    """
    Process a simulation output file to extract parameter information and metrics.
    
    Args:
        input_file (str): Path to the simulation output file
        output_file (str, optional): Path to save the CSV output (if out_csv=True)
        out_csv (bool): Whether to save results as CSV (True) or return data structure (False)
    
    Returns:
        Union[List[Dict[str, Any]], None]: List of parameter blocks if out_csv=False, otherwise None
        
    Notes:
        The function parses the output file to extract parameter blocks containing
        configuration parameters and performance metrics (accuracy, ranks, pool counts).
    """
    # Read the entire file
    try:
        with open(input_file, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: Could not find input file '{input_file}'")
        return [] if not out_csv else None
    
    blocks = []
    block_data = {}
    in_block = False
    
    # Process each line
    for line in lines:
        line = line.strip()
        
        # Start of a new parameter block
        if line.startswith('***********params************'):
            if in_block and block_data:
                blocks.append(block_data.copy())
            block_data = {}
            in_block = True
            continue
        
        # End of all parameters - check for final analysis or another params block
        if "Final analysis" in line or (line.startswith('***********params************') and in_block):
            if in_block and block_data:
                blocks.append(block_data.copy())
            if "Final analysis" in line:
                break
            # If it's another params block, continue processing
            if line.startswith('***********params************'):
                block_data = {}
                in_block = True
            continue
        
        # Extract the final metrics line with all values
        if in_block and ('accuracy:' in line and 'coop_accuracy:' in line and 'mean sensitivity:' in line 
                         and 'avg rank:' in line and 'total pools:' in line):
            # Parse the complex metrics line
            # Format: accuracy: 1.0; coop_accuracy: 1.0; mean sensitivity: 0.93375; avg rank: 0.0; stdev rank: 0.0; total pools: 9998.09
            
            accuracy_match = re.search(r'accuracy: ([0-9]*\.?[0-9]+)', line)
            coop_acc_match = re.search(r'coop_accuracy: ([0-9]*\.?[0-9]+)', line)
            mean_sens_match = re.search(r'mean sensitivity: ([0-9]*\.?[0-9]+)', line)
            avg_rank_match = re.search(r'avg rank: ([0-9]*\.?[0-9]+)', line)
            stdev_rank_match = re.search(r'stdev rank: ([0-9]*\.?[0-9]+)', line)
            total_pools_match = re.search(r'total pools: ([0-9]*\.?[0-9]+)', line)
            
            if accuracy_match:
                block_data['accuracy'] = float(accuracy_match.group(1))
            if coop_acc_match:
                block_data['coop_accuracy'] = float(coop_acc_match.group(1))
            if mean_sens_match:
                block_data['mean_sensitivity'] = float(mean_sens_match.group(1))
            if avg_rank_match:
                block_data['mean_rank'] = float(avg_rank_match.group(1))
            if stdev_rank_match:
                block_data['stdev_rank'] = float(stdev_rank_match.group(1))
            if total_pools_match:
                block_data['num_pools'] = float(total_pools_match.group(1))
            
            # This completes the block
            blocks.append(block_data.copy())
            in_block = False
            continue
                
        # Extract parameter key-value pairs within the params block
        if in_block and ': ' in line and not line.startswith('Best ranks'):
            key, value = line.split(': ', 1)
            
            # Convert values to appropriate types
            if key in ['pool_size', 'num_meet', 'num_redun', 'final_size', 'num_top']:
                try:
                    block_data[key] = int(value)
                except ValueError:
                    block_data[key] = value
            elif key in ['frac_top']:
                try:
                    block_data[key] = float(value)
                except ValueError:
                    block_data[key] = value
            elif key == 'tiebreak_round':
                block_data[key] = value.lower() == 'true'
            else:
                block_data[key] = value
            continue
    
    # Handle any remaining block
    if in_block and block_data:
        blocks.append(block_data.copy())
    
    # If not returning CSV, return the blocks as-is
    if not out_csv:
        return blocks
    
    # If saving as CSV, create and sort dataframe
    if blocks:
        df = pd.DataFrame(blocks)
        
        # Reorder columns to match desired output format
        desired_columns = ['pool_size', 'num_meet', 'num_redun', 'frac_top', 'final_size', 
                          'target_metric', 'tiebreak_round', 'num_top', 'score_mode',
                          'accuracy', 'coop_accuracy', 'mean_sensitivity', 'mean_rank', 
                          'stdev_rank', 'num_pools']
        
        # Only include columns that exist in the dataframe
        available_columns = [col for col in desired_columns if col in df.columns]
        if available_columns:
            df = df[available_columns]
        
        # Sort by accuracy (descending), then by mean_rank (ascending), then by num_pools (ascending)
        sort_columns = []
        if 'accuracy' in df.columns:
            sort_columns.append(('accuracy', False))
        if 'mean_rank' in df.columns:
            sort_columns.append(('mean_rank', True))
        if 'num_pools' in df.columns:
            sort_columns.append(('num_pools', True))
            
        if sort_columns:
            df.sort_values(
                by=[col for col, _ in sort_columns],
                ascending=[asc for _, asc in sort_columns],
                inplace=True
            )
        
        # Save to CSV if output file specified
        if output_file:
            try:
                df.to_csv(output_file, index=False)
                print(f"Results saved to {output_file}")
            except Exception as e:
                print(f"Error saving results to CSV: {e}")
    else:
        print("Warning: No valid data blocks found in the input file")
    
    return None


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python process_output.py <input_file> [output_file]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = f'{input_file}_extract.csv' if len(sys.argv) < 3 else sys.argv[2]
    
    process_out(input_file, output_file)