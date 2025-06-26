#!/usr/bin/env python

"""
Covering Design Retrieval Module

This module provides functionality to query and retrieve covering designs from 
the La Jolla Covering Repository (LJCR). Covering designs are combinatorial objects
where a set of k-element subsets (blocks) cover all t-element subsets of a v-element set.

The module caches query results to avoid redundant network requests.

Usage:
    blocks = query_ljcr(v, k, t)
    where:
        v: Size of the ground set
        k: Size of each block
        t: Size of subsets to be covered
"""

from bs4 import BeautifulSoup as bs
import requests
from functools import lru_cache

@lru_cache(maxsize=128)
def query_ljcr(v, k, t):
    """
    Query covering designs from ljcr.dmgordon.org and parse the result.
    
    Args:
        v (int): Number of elements in the set
        k (int): Size of each block
        t (int): Coverage parameter
        
    Returns:
        list: List of blocks representing the covering design
    """
    url = f'https://ljcr.dmgordon.org/cover/get_cover.php?v={v}&k={k}&t={t}'
    response = requests.get(url)
    
    soup = bs(response.text, 'html.parser')
    pre_tag = soup.find('pre')
    
    if not pre_tag:
        return []
        
    # Parse the text inside the <pre> tag into blocks of numbers
    blocks = []
    for line in pre_tag.text.strip().split('\n'):
        if line.strip():
            try:
                block = [int(num) for num in line.split()]
                if block:
                    blocks.append(block)
            except ValueError:
                # Skip lines that don't contain valid integers
                continue
    
    return blocks


if __name__ == "__main__":
    # Fixed typo in function name
    print(f"Covering design (7,3,2): {query_ljcr(7, 3, 2)}")
    print(f"Covering design (7,3,3): {query_ljcr(7, 3, 3)}")