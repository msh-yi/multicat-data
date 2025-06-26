import numpy as np
import basico as cps
import re
import sys
import yaml
from datetime import datetime
import csv
import os
import coop_solver
import uuid
from scipy.integrate import odeint, solve_ivp
import pandas as pd
from numba import jit, float64, int64

IRREV_K = 1E-9

class Reaction:
    """
    Base class for simulating reactions with catalysts that may exhibit cooperative effects.
    
    This class provides a framework for modeling reactions with multiple catalysts where
    inter-catalyst interactions may enhance or inhibit reaction rates.
    
    Attributes:
        cats (list): A list of integers that indicates which catalysts (by index) are in this reaction
        num_cats (int): Number of catalysts in the reaction
        rxn_time (float): Reaction time in arbitrary units
        neg_strength (float): Class variable for negative cooperativity strength
        coop_mech (str): Class variable defining the cooperativity mechanism ('dimer' or 'multistep')
    
    Assumptions:
        - Inter-catalyst interactions are infinitely faster than the main reaction
        - Binding events are irreversible
        - All binding strengths are equal
        - Total catalyst loading is constant (evenly distributed among catalysts)
    """

    #defaults
    neg_strength = None
    coop_mech = None # we assume that we won't be mixing dimer and multistep modes

    @staticmethod
    def create(*args, **kwargs):
        if DimerReaction.coop_mech == 'dimer':
            return DimerReaction(*args, **kwargs)
        if DimerReaction.coop_mech == 'multistep':
            return MultistepReaction(*args, **kwargs)
    

    def __init__(self, cats, rxn_time=0.05):
        self.cats = cats
        self.num_cats = len(cats)
        self.rxn_time = rxn_time

    def equil_cats(self):
        pass

    def calculate_k_eff(self):
        pass

    def run_reaction(self):
        pass

    @classmethod
    def catalyst_generator(cls, *args, **kwargs):
        pass

    @classmethod
    def coop_generator(cls, *args, **kwargs):
        pass
    
class DimerReaction(Reaction):
    '''
    Base class for simulating reactions with catalysts that may exhibit cooperative effects.
    
    This class provides a framework for modeling reactions with multiple catalysts where
    inter-catalyst interactions may enhance or inhibit reaction rates.
    
    Attributes:
        cats (list): A list of integers that indicates which catalysts (by index) are in this reaction
        num_cats (int): Number of catalysts in the reaction
        rxn_time (float): Reaction time in arbitrary units
        neg_strength (float): Class variable for negative cooperativity strength
        coop_mech (str): Class variable defining the cooperativity mechanism ('dimer' or 'multistep')
    
    Assumptions:
        - Inter-catalyst interactions are infinitely faster than the main reaction
        - Binding events are irreversible
        - All binding strengths are equal
        - Total catalyst loading is constant (evenly distributed among catalysts)

    class variables:
        - cat_rate_dict: a dict of catalysts and rates
        - coop_dicts: a list of length two, containing first the dict for negative cooperativity and then the dict for positive cooperativity
    
    '''

    def __init__(self, cats, rxn_time=0.05):
        super().__init__(cats, rxn_time)
        if self.cats:
            self.cat_species_concs = self.equil_cats() # get concentrations of all active cats after equilibrating

            self.k_eff = self.calculate_k_eff()
        else:
            self.k_eff = 0

        self.output = self.k_eff

    @staticmethod
    @jit(float64[:](float64, float64[:], int64[:], int64[:], float64[:], float64[:], int64[:]), nopython=True)
    def odes(t, y, reaction_i, reaction_j, reaction_k1, reaction_k2, index_map):
        """
        Numba-accelerated ODE function for dimer equilibrium calculations.
        
        Parameters:
            t (float): Current time
            y (ndarray): Current concentrations
            reaction_i (ndarray): Indices of first catalyst in each reaction
            reaction_j (ndarray): Indices of second catalyst in each reaction
            reaction_k1 (ndarray): Forward rate constants for each reaction
            reaction_k2 (ndarray): Reverse rate constants for each reaction
            index_map (ndarray): Mapping from catalyst indices to position in y array
            
        Returns:
            ndarray: Derivatives of each species concentration
        """
        dydt = np.zeros_like(y)
        for idx in range(len(reaction_i)):
            i, j = reaction_i[idx], reaction_j[idx]
            k1, k2 = reaction_k1[idx], reaction_k2[idx]
            c_i_idx = index_map[i]
            c_j_idx = index_map[j]
            c_ij_idx = index_map[i * 1000 + j]  # Assuming max 1000 catalysts
            dydt[c_i_idx] -= k1 * y[c_i_idx] * y[c_j_idx] - k2 * y[c_ij_idx]
            dydt[c_j_idx] -= k1 * y[c_i_idx] * y[c_j_idx] - k2 * y[c_ij_idx]
            dydt[c_ij_idx] = k1 * y[c_i_idx] * y[c_j_idx] - k2 * y[c_ij_idx]
        return dydt

    def equil_cats_numba(self):
        """
        WARNING: NOT IN USE
        Calculate equilibrium concentrations of catalyst species using numba-optimized ODE solver.
        
        This method is more efficient than the standard approach but is currently not in use.
        
        Returns:
            pd.DataFrame: Equilibrium concentrations of all catalyst species
        """
        irrev_k = 1E-12

        def set_up_eqm(irrev_k):
            # Dict of initial concs
            species_map = {f'c{idx}': 1/len(self.cats) for idx in self.cats}
            
            # Combine positive and negative cooperation dictionaries
            all_coop_pairs = set(DimerReaction.pos_coop_dict.keys()) | set(DimerReaction.neg_coop_dict.keys())
            
            # Only create dimer species for pairs in cooperation dictionaries
            for i, j in all_coop_pairs:
                if i in self.cats and j in self.cats and i < j:
                    species_map[f'c{i}_c{j}'] = 0

            index_map = {spec: i for i, spec in enumerate(species_map)}
            y0 = np.array(list(species_map.values()))

            reactions = []
            if DimerReaction.neg_coop_dict:
                for k, v in DimerReaction.neg_coop_dict.items():
                    if k[0] in self.cats and k[1] in self.cats and k[0] != k[1]:
                        reactions.append((k, (DimerReaction.neg_strength * 0.1, irrev_k)))

            if DimerReaction.pos_coop_dict:
                for k, v in DimerReaction.pos_coop_dict.items():
                    if k[0] in self.cats and k[1] in self.cats and k[0] != k[1]:
                        reactions.append((k, (0.1, irrev_k)))  # Irreversible reaction

            # Prepare data for Numba-optimized odes
            reaction_i = np.array([k[0] for k, _ in reactions], dtype=np.int64)
            reaction_j = np.array([k[1] for k, _ in reactions], dtype=np.int64)
            reaction_k1 = np.array([v[0] for _, v in reactions], dtype=np.float64)
            reaction_k2 = np.array([v[1] for _, v in reactions], dtype=np.float64)
            
            # Convert index_map to a Numba-compatible format
            max_cat = max(self.cats)
            numba_index_map = np.full((max_cat + 1) * 1001, -1, dtype=np.int64)
            for k, v in index_map.items():
                if k.startswith('c'):
                    if '_' in k:
                        i, j = map(int, k[1:].split('_c'))
                        numba_index_map[i * 1000 + j] = v
                    else:
                        numba_index_map[int(k[1:])] = v

            params = {
                'reactions': reactions,
                'index_map': index_map,
                'reaction_i': reaction_i,
                'reaction_j': reaction_j,
                'reaction_k1': reaction_k1,
                'reaction_k2': reaction_k2,
                'numba_index_map': numba_index_map
            }

            def numba_odes(t, y):
                return DimerReaction.odes(t, y, reaction_i, reaction_j, reaction_k1, reaction_k2, numba_index_map)

            sol = solve_ivp(numba_odes, [0, 200], y0, method='RK45', rtol=1e-6, atol=1e-12)
            
            return sol, params
        
        def check_equilibrium(sol, params, tolerance=1e-6):
            final_rates = np.abs(DimerReaction.odes(sol.t[-1], sol.y[:, -1], 
                                                params['reaction_i'], params['reaction_j'], 
                                                params['reaction_k1'], params['reaction_k2'], 
                                                params['numba_index_map']))
            return np.all(final_rates < tolerance)

        def solve_with_adaptive_time(irrev_k, max_time=1e6, tolerance=1e-6):
            sol, params = set_up_eqm(irrev_k)
            
            while sol.t[-1] < max_time:
                if check_equilibrium(sol, params, tolerance):
                    return sol, params
                
                new_end_time = min(sol.t[-1] * 2, max_time)
                
                def numba_odes(t, y):
                    return DimerReaction.odes(t, y, params['reaction_i'], params['reaction_j'], 
                                        params['reaction_k1'], params['reaction_k2'], 
                                        params['numba_index_map'])
                
                sol = solve_ivp(numba_odes, [sol.t[-1], new_end_time], sol.y[:, -1], 
                                method='RK45', rtol=1e-6, atol=1e-12)
            
            return sol, params

        sol, params = solve_with_adaptive_time(irrev_k)

        while not sol.success:
            irrev_k *= 10
            if irrev_k > 0.1:
                raise ValueError("irrev_k is too big")
            sol, params = solve_with_adaptive_time(irrev_k)

        conc_dict = {spec: sol.y[i, -1] for spec, i in params['index_map'].items()}

        return pd.DataFrame.from_dict(conc_dict, orient='index', columns=['concentration'])
    
    def equil_cats(self):
        """
        Calculate equilibrium concentrations of catalyst species using scipy's ODE solver.
        
        Sets up and solves the system of ODEs representing the equilibrium between
        free catalysts and their dimers based on cooperativity parameters.
        
        Returns:
            pd.DataFrame: Equilibrium concentrations of all catalyst species
        """
        irrev_k = 1E-12

        def odes(t, y, params):
            dydt = np.zeros(len(y))
            for (i, j), (k1, k2) in params['reactions']:
                c_i_idx = params['index_map'][f'c{i}']
                c_j_idx = params['index_map'][f'c{j}']
                c_ij_idx = params['index_map'][f'c{i}_c{j}']
                dydt[c_i_idx] -= k1 * y[c_i_idx] * y[c_j_idx] - k2 * y[c_ij_idx]
                dydt[c_j_idx] -= k1 * y[c_i_idx] * y[c_j_idx] - k2 * y[c_ij_idx]
                dydt[c_ij_idx] = k1 * y[c_i_idx] * y[c_j_idx] - k2 * y[c_ij_idx]
            return dydt

        def set_up_eqm(irrev_k):
            # Dict of initial concs
            species_map = {f'c{idx}': 1/len(self.cats) for idx in self.cats}
            
            # Combine positive and negative cooperation dictionaries
            all_coop_pairs = set(DimerReaction.pos_coop_dict.keys()) | set(DimerReaction.neg_coop_dict.keys())
            
            # Only create dimer species for pairs in cooperation dictionaries
            for i, j in all_coop_pairs:
                if i in self.cats and j in self.cats and i < j:
                    species_map[f'c{i}_c{j}'] = 0

            index_map = {spec: i for i, spec in enumerate(species_map)}
            y0 = np.array(list(species_map.values()))

            reactions = []
            if DimerReaction.neg_coop_dict:
                for k, v in DimerReaction.neg_coop_dict.items():
                    if k[0] in self.cats and k[1] in self.cats and k[0] != k[1]:
                        reactions.append((k, (DimerReaction.neg_strength * 0.1, irrev_k)))

            if DimerReaction.pos_coop_dict:
                for k, v in DimerReaction.pos_coop_dict.items():
                    if k[0] in self.cats and k[1] in self.cats and k[0] != k[1]:
                        reactions.append((k, (0.1, irrev_k)))  # Irreversible reaction

            params = {
                'reactions': reactions,
                'index_map': index_map
            }
            #print(f"Solving ODE system with initial y0: {y0} and params: {params}")
            
            sol = solve_ivp(odes, [0, 200], y0, args=(params,), method='RK45', rtol=1e-6, atol=1e-12)
            # for now we fox at 200

            #print(f"Solution status: {sol.success}, Message: {sol.message}")
            #print(f"y: {sol.y}")
            
            return sol, params
        
        def check_equilibrium(sol, params, tolerance=1e-6):
            # Check if the rate of change for all species is below the tolerance
            final_rates = np.abs(odes(sol.t[-1], sol.y[:, -1], params))
            return np.all(final_rates < tolerance)

        def solve_with_adaptive_time(irrev_k, max_time=1e6, tolerance=1e-6):
            sol, params = set_up_eqm(irrev_k)
            
            while sol.t[-1] < max_time:
                if check_equilibrium(sol, params, tolerance):
                    #print(f"Equilibrium reached at t = {sol.t[-1]}")
                    return sol, params
                
                # If not at equilibrium, continue integration
                new_end_time = min(sol.t[-1] * 2, max_time)  # Double the integration time
                sol = solve_ivp(odes, [sol.t[-1], new_end_time], sol.y[:, -1], args=(params,), method='RK45', rtol=1e-6, atol=1e-12)
            
            #print(f"Warning: Max time ({max_time}) reached without achieving equilibrium")
            return sol, params

        sol, params = solve_with_adaptive_time(irrev_k)

        while not sol.success:
            irrev_k *= 10
            if irrev_k > 0.1:
                raise ValueError("irrev_k is too big")
            sol, params = solve_with_adaptive_time(irrev_k)

        #print(sol)
        conc_dict = {spec: sol.y[i, -1] for spec, i in params['index_map'].items()}

        return pd.DataFrame.from_dict(conc_dict, orient='index', columns=['concentration'])

    def equil_cats_cps(self):
        """
        WARNING: NOT ACTIVELY USED
        Calculate equilibrium concentrations of catalyst species using COPASI (via basico).
        
        This alternative implementation uses the COPASI backend for steady-state calculations.
        
        Returns:
            pd.DataFrame: Equilibrium concentrations of all catalyst species
        """

        irrev_k = 1E-12
        return_status = 0

        def set_up_eqm(irrev_k):
            cps.new_model(name=f'equil_{uuid.uuid4()}')
            for idx in self.cats:
                cps.add_reaction(f'c{idx}s', f'c{idx} -> c{idx}') # s for self
                cps.set_species(f'c{idx}', initial_concentration=1/self.num_cats) # if we have two cats then half each

            if DimerReaction.neg_coop_dict:
                #stick all the negatively cooperative catalysts together
                for k in DimerReaction.neg_coop_dict:
                    if (k[0] in self.cats) and (k[1] in self.cats): # a cat pair applies here!
                        rxn_name = f'c{k[0]}-c{k[1]}b'
                        cps.add_reaction(rxn_name, f'c{k[0]} + c{k[1]} = c{k[0]}_c{k[1]}') # b for bind
                        cps.set_species(f'c{k[0]}_c{k[1]}', initial_concentration=0)
                        cps.set_reaction_parameters(f'({rxn_name}).k2', value=irrev_k) # irreversible for now
                        cps.set_reaction_parameters(f'({rxn_name}).k1', value=DimerReaction.neg_strength * 0.1) # irreversible for now

            if DimerReaction.pos_coop_dict:    
                for k in DimerReaction.pos_coop_dict:
                    if (k[0] in self.cats) and (k[1] in self.cats): # a cat pair applies here!
                        rxn_name = f'c{k[0]}-c{k[1]}b'
                        cps.add_reaction(rxn_name, f'c{k[0]} + c{k[1]} = c{k[0]}_c{k[1]}')
                        cps.set_species(f'c{k[0]}_c{k[1]}', initial_concentration=0)
                        cps.set_reaction_parameters(f'({rxn_name}).k2', value=irrev_k) #irrev for now
        
            #print(cps.get_reaction_parameters())
            #print(cps.get_species())    
            return_status = cps.run_steadystate()

            return return_status

        return_status = set_up_eqm(irrev_k)

        while not return_status:
            irrev_k *= 10
            #print(f"{self.cats}: irrev_k not large enough, increased to {irrev_k}")
            if irrev_k > 0.1:
                #print("irrev_k is too big")
                os._exit()
                return
            return_status = set_up_eqm(irrev_k)

        conc_dict = cps.get_species()[['concentration']]
        #print(conc_dict)

        return conc_dict
        
        #TODO: In the future we'll figure out reasonable cat concentrations and/or k_cats to match reality

    def calculate_k_eff(self):
        """
        Calculate the effective rate constant based on catalyst species concentrations.
        
        Computes a weighted sum of rate constants based on catalyst concentrations and
        cooperativity factors for dimers.
        
        Returns:
            float: The effective rate constant for the reaction
        """
        k_eff = 0
        for cat, row in self.cat_species_concs.iterrows():
            concentration = row['concentration']
            if concentration > 1e-10:  # Ignore very small concentrations
                cat_species = tuple(map(int, re.findall(r'\d+', cat)))
                if len(cat_species) == 1:
                    k_eff += concentration * DimerReaction.cat_rate_dict[cat_species[0]]
                else:  # a set of cats
                    k_set_composite = sum([DimerReaction.cat_rate_dict[cpt] for cpt in cat_species])
                    sorted_species = tuple(sorted(cat_species))  # Ensure consistent ordering
                    if sorted_species in DimerReaction.pos_coop_dict:
                        k_set = k_set_composite * DimerReaction.pos_coop_dict[sorted_species]
                    elif sorted_species in DimerReaction.neg_coop_dict:
                        k_set = k_set_composite * DimerReaction.neg_coop_dict[sorted_species]
                    else:
                        raise KeyError(f'Catalytic species {sorted_species} not found in coop dict')
                    k_eff += concentration * k_set
        
        return k_eff

    def run_rxn(self):
        """
        Simulate the reaction using scipy's ODE solver.
        
        Models the conversion of substrate A to product B using the calculated effective
        rate constant.
        
        Returns:
            float: Final yield of product B
        """

        def rxn(t, y, k_eff):
            return [-k_eff * y[0], k_eff * y[0]]

        y0 = [1, 0]  # initial concentrations of A and B
        try:
            sol = solve_ivp(rxn, [0, self.rxn_time], y0, args=(self.k_eff,), method='LSODA')
            if sol.success:
                yld = sol.y[1, -1]  # concentration of B at the end of rxn_time
                return yld
            else:
                raise RuntimeError(f"ODE solver failed: {sol.message}")
        except Exception as e:
            print(f"Error in run_rxn: {str(e)}")
            return None

    def run_rxn_cps(self):
        """
        NOT ACTIVELY USED
        Simulate the reaction using scipy's ODE solver.
        
        Models the conversion of substrate A to product B using the calculated effective
        rate constant.
        
        Returns:
            float: Final yield of product B
        """
        cps.new_model(name = 'main')
        cps.add_reaction(f'main', 'A -> B') # s for self
        cps.set_species('A', initial_concentration=1)
        cps.set_species('B', initial_concentration=0)

        cps.set_reaction_parameters(f'(main).k1', value=self.k_eff)
        timecourse = cps.run_time_course(method='deterministic', duration=self.rxn_time)
        df_endpoint = cps.get_species()
        yld = df_endpoint.loc['B']['concentration']

        return yld
    
    @classmethod
    def catalyst_generator(cls, out, N, seed=50, mean_rate=10, sd_rate=10, rate_outlier_prob=0.1, **kwargs):
        """
        Generate a population of catalysts with varied rate constants.
        
        Parameters:
            out (str): Prefix for output file path
            N (int): Number of catalysts to generate
            seed (int): Random number generator seed
            mean_rate (float): Mean value for catalyst rate constants
            sd_rate (float): Standard deviation for catalyst rate constants
            rate_outlier_prob (float): Probability of generating outlier catalysts
            **kwargs: Additional parameters
            
        Side effects:
            - Sets cls.cat_rate_dict
            - Writes catalyst rates to a CSV file
        """
        rng = np.random.default_rng(seed=seed)

        # Generate the rates using a normal distribution
        rates = rng.normal(loc=mean_rate, scale=sd_rate, size=N)
        # Introduce outliers into the distribution
        num_outliers = int(rate_outlier_prob * len(rates))
        outlier_indices = rng.choice(len(rates), num_outliers, replace=False)
        outliers = rng.normal(loc=mean_rate * 2, scale=sd_rate, size=num_outliers) 
        # more knobs to turn here, potentially
        rates[outlier_indices] = outliers

        # Clip the values between 0 and 100
        rates = np.clip(rates, 0, 100)
        indexes = np.arange(N)
        # Create list of Catalyst objects
        #cat_list = [Catalyst(rate, index) for rate, index in zip(rates, indexes)]
        cls.cat_rate_dict = {index: rate for index, rate in zip(indexes, rates)}
        with open(f'{out}_cat_rate.csv', 'w', newline='') as csvfile:
            # Create a writer object
            # The fieldnames parameter is a sequence of keys that identify the order in which values in the dictionary can be written to the CSV file.
            writer = csv.writer(csvfile)

            # Write the data
            for k,v in cls.cat_rate_dict.items():
                writer.writerow([k, v])


    @classmethod
    def coop_generator(cls, out, **kwargs):
        """
        Generate cooperativity matrices for positive and negative interactions.
        
        Parameters:
            out (str): Prefix for output file path
            **kwargs: Parameters including seed, p_pos, range_pos, p_neg, range_neg
            
        Side effects:
            - Sets cls.pos_coop_dict and cls.neg_coop_dict
            - Writes cooperativity values to a CSV file
        """
        outlier_indices = [] # to avoid clashes
        rng = np.random.default_rng(seed=kwargs['seed'])
        
        def gen_pos_coop_mat(N, p_pos=0.01, range_pos=5, **kwargs):
            # Create a matrix with values from a normal distribution centered around 1
            # Create a N*N matrix with ones in the upper triangle
            matrix = np.triu(np.ones((N, N)))

            # Calculate the number of outliers
            num_outliers = int(np.triu(np.ones((N, N))).sum() * p_pos)

            # Generate random indices for the outliers
            indices = np.triu_indices(N)
            global outlier_indices 
            outlier_indices = rng.choice(len(indices[0]), num_outliers, replace=False)
            #print(outlier_indices)
            # Replace the selected indices with random numbers between 1 and range
            matrix[indices[0][outlier_indices], indices[1][outlier_indices]] = rng.choice(np.arange(2, range_pos + 1), num_outliers, replace=True)
            
            # np.fill_diagonal(matrix, 1)
            
            pos_coop_dict_loc = {}
            for i,j in zip(indices[0][outlier_indices], indices[1][outlier_indices]):
                # need to remove accidental cat_i and cat_i coop
                #if i != j: # allow self dimerization
                pos_coop_dict_loc[(i,j)] = matrix[i,j]
            
            #cls.pos_coop_mat = matrix
            cls.pos_coop_dict = pos_coop_dict_loc

        def gen_neg_coop_mat(N, p_neg=0.05, range_neg=0.3, **kwargs):
            # Create a matrix with values from a normal distribution centered around 1

            # Create a N*N matrix with ones in the upper triangle
            matrix = np.triu(np.ones((N, N)))

            # Calculate the number of outliers
            num_outliers = int(np.triu(np.ones((N, N))).sum() * p_neg)

            # Generate random indices for the outliers
            indices = np.triu_indices(N)
            # include only pairs that dont have pos coop
            global outlier_indices
            #print(outlier_indices)
            choices = np.array([i for i in np.arange(len(indices[0])) if i not in outlier_indices])

            neg_outlier_indices = rng.choice(choices, num_outliers, replace=False)
            # Replace the selected indices with random numbers between 0 and neg_coop
            matrix[indices[0][neg_outlier_indices], indices[1][neg_outlier_indices]] = rng.uniform(0, range_neg, num_outliers)
            np.fill_diagonal(matrix, 1)

            neg_coop_dict_loc = {}
            for i,j in zip(indices[0][neg_outlier_indices], indices[1][neg_outlier_indices]):
                if i != j:
                    neg_coop_dict_loc[(i,j)] = matrix[i,j]
            
            #cls.neg_coop_mat = matrix
            cls.neg_coop_dict = neg_coop_dict_loc

        gen_pos_coop_mat(**kwargs)
        gen_neg_coop_mat(**kwargs)

        with open(f'{out}_coop_mat.csv', 'w', newline='') as csvfile:
            # Create a writer object
            # The fieldnames parameter is a sequence of keys that identify the order in which values in the dictionary can be written to the CSV file.
            writer = csv.writer(csvfile)

            # Write the data
            for k, v in cls.pos_coop_dict.items():
                writer.writerow([k, v]) 
            for k, v in cls.neg_coop_dict.items():
                writer.writerow([k, v])

class MultistepReaction(Reaction):
    """
    DEPRECATED: USE AT OWN RISK!
    Models a reaction with multiple sequential steps catalyzed by different catalysts.
    
    This class extends Reaction to simulate multi-step reactions where different catalysts
    have varying affinities for each step.
    
    Attributes:
        cats (list): List of catalyst indices involved in the reaction
        rxn_time (float): Reaction time in arbitrary units
        cat_species_concs (dict): Equilibrium concentrations of catalysts
        k_eff (list): List of effective rate constants for each reaction step
        yld (float): Final yield of product
        output (float): The calculated yield
        
    Class Variables:
        num_steps (int): Number of steps in the reaction pathway
        cat_rate_dict (dict): Dictionary mapping catalyst indices to their intrinsic rates for each step
        coop_dict (dict): Dictionary mapping catalyst indices to their binding affinities for each step
    """
    num_steps = None

    def __init__(self, cats, rxn_time=1):
        super().__init__(cats, rxn_time)
        if self.cats:
            self.cat_species_concs = self.equil_cats() # get concentrations of all active cats after equilibrating
            self.k_eff = self.calculate_k_eff()
            self.yld = self.run_reaction()
            self.output = self.yld
            # k_eff here is a list, not a float!
        else:
            self.k_eff = 0
            self.yld = 0
            self.output =0
            

    def equil_cats(self):
        # unlike dimerreaction, returns a dict where the index is already an int
        conc = 1/self.num_cats
        return {i: conc for i in self.cats}
    
    def calculate_k_eff(self):
        # for each step weight k_eff of each cat by its concentration and binding affinity

        self.k_eff = [0] * MultistepReaction.num_steps

        for cat in self.cat_species_concs.keys(): # cat is the specific index
            for i in range(MultistepReaction.num_steps):
                self.k_eff[i] += self.cat_species_concs[cat] * MultistepReaction.cat_rate_dict[cat][i] * MultistepReaction.coop_dict[cat][i]
            #for each step, let all catalysts compete to bind int 

        return self.k_eff


    def run_reaction(self):
        cps.new_model(name = 'main')
        for i in range(MultistepReaction.num_steps):
            # The first step: A -> int0
            if i == 0:
                cps.add_reaction(f'step{i}', f'A -> int{i}')
            # The last step: int(num_steps-2) -> B
            elif i == MultistepReaction.num_steps - 1:
                cps.add_reaction(f'step{i}', f'int{i-1} -> B')
                cps.set_species(f'int{i-1}', initial_concentration=0)
            # Intermediate steps: int(i-1) -> int(i)
            else:
                cps.add_reaction(f'step{i}', f'int{i-1} -> int{i}')
                cps.set_species(f'int{i-1}', initial_concentration=0)
            
            # Set the reaction parameters for each step
            cps.set_reaction_parameters(f'(step{i}).k1', value=self.k_eff[i])
                
        cps.set_species('A', initial_concentration=1)
        cps.set_species('B', initial_concentration=0)
        #print(cps.get_reaction_parameters())
        
        timecourse = cps.run_time_course(method='deterministic', duration=self.rxn_time)
        df_endpoint = cps.get_species()
        yld = df_endpoint.loc['B']['concentration']
        return yld

    @classmethod
    def catalyst_generator(cls, out, N, seed=50, mean_rate=10, sd_rate=10, rate_outlier_prob=0.1, **kwargs):
        ## generate a tuple depending on how many steps
        rng = np.random.default_rng(seed=seed)

        # Generate the rates using a normal distribution
        rates = rng.normal(loc=mean_rate, scale=sd_rate, size=(N, cls.num_steps))
        # Introduce outliers into the distribution
        num_outliers = int(rate_outlier_prob * len(rates))
        outlier_indices = rng.choice(len(rates), num_outliers, replace=False)
        outliers = rng.normal(loc=mean_rate * 2, scale=sd_rate, size=(num_outliers, cls.num_steps)) 
        # more knobs to turn here, potentially
        rates[outlier_indices] = outliers

        # Clip the values between 0 and 100
        rates = np.clip(rates, 0, 100)
        indexes = np.arange(N)
        # Create list of Catalyst objects
        #cat_list = [Catalyst(rate, index) for rate, index in zip(rates, indexes)]
        cls.cat_rate_dict = {index: tuple(rate) for index, rate in zip(indexes, rates)}
        with open(f'{out}_cat_rate.csv', 'w', newline='') as csvfile:
            # Create a writer object
            # The fieldnames parameter is a sequence of keys that identify the order in which values in the dictionary can be written to the CSV file.
                writer = csv.writer(csvfile)

                # Write the data
                for k,v in cls.cat_rate_dict.items():
                    writer.writerow([k, v])

    @classmethod
    def coop_generator(cls, N, out, seed, coop_means, coop_corrs, coop_vars, **kwargs):
        '''
        corrs: a list [0.5, 0.3, 0.2] means first and second, first and third, second and third
        
        
        '''
        k = cls.num_steps # number of distributions to make
        rng = np.random.default_rng(seed=seed+2)
        cov = np.ones((k, k))
        cov[np.triu_indices(k, 1)] = coop_corrs
        cov[np.tril_indices(k, -1)] = coop_corrs
        np.fill_diagonal(cov, coop_vars)
        # between 0 and 1

        affinities = rng.multivariate_normal(coop_means, cov, size=N)
        affinities = np.clip(affinities, 0, 1)
        # scale so that affinities for each step sum to 1
        col_sums = affinities.sum(axis=0)
        affinities = affinities/col_sums
        # no outliers for now
        cls.coop_dict = {index: tuple(affinity) for index, affinity in zip(range(N), affinities)}
        
        with open(f'{out}_coop_mat.csv', 'w', newline='') as csvfile:
        # Create a writer object
        # The fieldnames parameter is a sequence of keys that identify the order in which values in the dictionary can be written to the CSV file.
            writer = csv.writer(csvfile)

            # Write the data
            for k,v in cls.coop_dict.items():
                writer.writerow([k, v])

def gen_landscape(config_file=None, config_dict=None, out=None):
    """
    Generate a reaction landscape based on configuration parameters.
    
    Creates catalysts and cooperativity parameters according to specified configuration,
    either from a file or a dictionary.
    
    Parameters:
        config_file (str, optional): Path to YAML configuration file
        config_dict (dict, optional): Configuration dictionary
        out (str): Prefix for output files
    
    Returns:
        None
    
    Raises:
        ValueError: If output path is not specified
    """
    if config_file:
        with open(config_file, 'r') as stream:
            try:
                config = yaml.safe_load(stream)
            except yaml.YAMLError as e:
                print(e)
                sys.exit(1)
    else:
        config = config_dict
    
    if not out:
        raise ValueError('need to specify output path!')
    
    rng = np.random.default_rng(seed=config['rng']['seed'])
    try:
        if config['simul']['mode'] == 'dimer':
            rxn_class = DimerReaction
        elif config['simul']['mode'] == 'multistep':
            rxn_class = MultistepReaction
            MultistepReaction.num_steps = config['simul']['num_steps']
    except KeyError as e:
        rxn_class = DimerReaction

    
    #if not specified, generate one landscape
    num_landscapes = config['simul'].get('num_landscapes', 1)

    for i in range(num_landscapes):
        out_i = out if num_landscapes == 1 else f'{out}_{i}'
        rxn_class.catalyst_generator(seed=rng.integers(0, 1E10), out=out_i, **config['simul'])
        rxn_class.coop_generator(seed=rng.integers(0, 1E10), out=out_i, **config['simul'])
    
    # if rxn_class == DimerReaction:
    #     return rxn_class.cat_rate_dict, rxn_class.pos_coop_dict, rxn_class.neg_coop_dict
    # elif rxn_class == MultistepReaction:
    #     print(f'num steps = {rxn_class.num_steps}')
    #     return rxn_class.cat_rate_dict, rxn_class.coop_dict
    return

def config(coop_mech=None, neg_strength=None):
    """
    Configure global simulation parameters.
    
    Parameters:
        coop_mech (str, optional): Cooperativity mechanism ('dimer' or 'multistep')
        neg_strength (float, optional): Strength of negative cooperativity
    """
    if neg_strength:
        Reaction.neg_strength = neg_strength
    if coop_mech:
        Reaction.coop_mech = coop_mech

def set_landscape(rate_file, coop_file, **kwargs):
    """
    Load pre-generated landscape parameters from files.
    
    Parameters:
        rate_file (str): Path to file containing catalyst rate constants
        coop_file (str): Path to file containing cooperativity parameters
        **kwargs: Additional parameters
        
    Returns:
        tuple: Loaded landscape parameters, format depends on cooperativity mechanism
        
    Raises:
        ValueError: If Reaction cooperativity mechanism is not specified
    """
    if Reaction.coop_mech == 'dimer':
        rxn_class = DimerReaction
        cat_rate_dict = coop_solver.csv_to_kv(rate_file)
        pos_coop_dict, neg_coop_dict = coop_solver.csv_to_kv(coop_file, coop=True)
        rxn_class.cat_rate_dict, rxn_class.pos_coop_dict, rxn_class.neg_coop_dict = cat_rate_dict, pos_coop_dict, neg_coop_dict

        return rxn_class.cat_rate_dict, rxn_class.pos_coop_dict, rxn_class.neg_coop_dict
    elif Reaction.coop_mech == 'multistep':
        rxn_class = MultistepReaction
        rxn_class.cat_rate_dict = coop_solver.csv_to_kv(rate_file)
        rxn_class.num_steps = len(next(v for v in rxn_class.cat_rate_dict.values()))
        rxn_class.coop_dict = coop_solver.csv_to_kv(coop_file)
        #print(f'num steps = {rxn_class.num_steps}')
        return rxn_class.cat_rate_dict, rxn_class.coop_dict
    else:
        ValueError('Reaction cooperativity mechanism not specified!')


