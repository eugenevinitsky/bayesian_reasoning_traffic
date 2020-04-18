"""Black box predictor of probability of pedestrian in a vehicle's grid cell using a pre-trained RL policy"""

import gym
import os
import sys
import time
import matplotlib.pyplot as plt
import numpy as np

import ray
try: from ray.rllib.agents.agent import get_agent_class
except ImportError: from ray.rllib.agents.registry import get_agent_class
from ray.tune.registry import register_env

def get_updated_priors(action, non_ped_obs, prev_priors, agent, policy_type="PPO"):
    """Black box predictor of the probability of pedestrian in the cells of veh id's FOV grid
    
    Parameters
    ----------
    veh_id: string
        vehicle veh_id whose FOV we're considering
    action: float
        actions taken by the vehicle is equivalent to the acceleration in m / s^2
    non_ped_obs: [non-ped-self-obs] + [other-veh-obs] * (num_max_obj - 1)
        self_obs: [yaw, speed, turn_num, edge_pos]
        other_veh_obs: ["rel_x", "rel_y", "speed", "yaw", "arrive_before"]
    prev_priors: dict (str : float)
        (pedestrian observation combination : probability)
        total of 2^6 pedestrian observation combinations and their corresponding updated prior probabilities
    agent: agent_object
        ray.rllib agent / policy object used to model the actions of vehicles  
        TODO(KL) Terminology - agent or policy? RL land :D
    policy_type: string
        type of policy used to model vehicle behavior
    
    Returns
    -------
    updated_prob_ped: [prob_ped_in_grid_1, ..., prob_ped_in_grid_6]
        Updated prior probabilties computed using a selected policy and via Bayes' rule
    new_priors: dict (str : float)
        (pedestrian observation combination : probability)
        total of 2^6 pedestrian observation combinations and their corresponding updated prior probabilities (given the action)
    """
    new_priors = {}
    pdf_joint_o_given_a_dct = {}
    self_obs = non_ped_obs[:4]
    other_veh_obs = non_ped_obs[4:]
    updated_prob_ped = []
    pure_prior = 1 / 2**6 # 'updated' prior probability of any ped obs combination for the first iteration
    cond_density_sum = 0 # sum of all conditional densities of obs combo given action

    len_6_bitstring_lst = make_permutations(num_digits=6, vals_per_dig=2)
    for ped_obs in len_6_bitstring_lst:
        modified_state = np.array(list(self_obs) + list(ped_obs) + list(other_veh_obs))
        mu, sigma = get_accel_gaussian_params(agent, modified_state, policy_type)
        # 1
        pdf_a_given_joint_o = prob_density(mu, sigma, action)
        # 2
        prior_pr_joint_o = prev_priors.get(ped_obs, pure_prior)
        # 3
        pdf_joint_o_given_a = pdf_a_given_joint_o * prior_pr_joint_o
        pdf_joint_o_given_a_dct[ped_obs] = pdf_joint_o_given_a
        cond_density_sum += pdf_joint_o_given_a

    for ped_obs in len_6_bitstring_lst:
        # 4
        pr_joint_o_given_a = pdf_joint_o_given_a_dct[ped_obs] / cond_density_sum
        new_priors[ped_obs] = pr_joint_o_given_a

    for grid_id in range(1, 7):
        # 5
        pr_o_given_a = 0
        for ped_combo in ped_combos_for_single_cond_pr(grid_id, 1):
            pr_o_given_a += new_priors[ped_combo]
        
        updated_prob_ped.append(pr_o_given_a)

    return updated_prob_ped, new_priors

    #############################
    #############################
    ###   Utility functions   ###
    #############################
    #############################

def prob_density(mu, sigma, acceleration):
    """Assuming acceleration comes from a Gaussian N(mu, sigma^2),
    compute the probability density of the given acceleration
    """
    coeff = 1 / np.sqrt(2 * np.pi * (sigma**2))
    exp = -0.5 * ((acceleration - mu) / sigma)**2
    return coeff * np.exp(exp)

def get_accel_gaussian_params(agent, modified_state, policy_type="PPO"):
    """Assuming the policy samples the output acceleration from a gaussian distribution, return the params for that gaussian.

    Parameters
    ----------
    agent: ray.rllib object
    modified_state: list of floats
        the state we're considering    
    policy_type: str
        the type of policy we're using to model vehicle behaviour
    
    Returns
    -------
    mu, sigma: float, float
    """
    if modified_state.dtype != 'float64':
        modified_state = modified_state.astype('float64')
    # TODO(KL) write this in a more general way ...
    if policy_type == "PPO":
        _, _, logit = agent.compute_action(modified_state, policy_id='av', full_fetch=True)
    else:
        raise NotImplementedError

    mu, ln_sigma = logit['behaviour_logits']
    sigma = np.exp(ln_sigma)
    return mu, sigma

# better name for this? 
def ped_combos_for_single_cond_pr(grid_idx, val, output_len=6):
    """Helper fn for computing a 'single' conditional probability e.g. p(o_3 = 1 | action)
    Returns a list of pedestrian combinations to sum over to get the single conditional probability.
    
    Params
    ------
    grid_idx: int from 1 to 6 representing the grid cell we're considering
    val: 0 or 1: 0 means no ped in the grid; 1 means ped in the grid
    
    3:0 means we want p(o_3 = 0 | a)
    Therefore, we can get the list of all possible length 5 bitstrings, and stitch '0' in the correct place.
    
    
    Returns
    -------
    list of bit strings of length 6
    """
    
    assert grid_idx >= 1 and grid_idx <= output_len
    res = []
    res_lst = make_permutations(output_len - 1, 2)
    
    for perm in res_lst:
        res.append(str(perm[:grid_idx - 1:] + str(val) + perm[grid_idx - 1:]))
        
    return res

def initial_prior_probs(num_digits=6, vals_per_dig=2):
    """Returns a dict with values of all permutations of bitstrings of length num_digits. 
    Each digit can take a value from 0 to (vals_per_dig - 1)"""
    uniform_prob = 1 / (vals_per_dig ** num_digits)
    res = make_dct_of_lsts(num_digits, vals_per_dig)
    for key in res.keys():
        res[key] = res[key] + [uniform_prob]
    return res

def make_dct_of_lsts(num_digits=6, vals_per_dig=2):
    """Return a dict with keys of bitstrings and values as empty lists. 
    Hardcoded for binary vals per var."""
    res = {}
    lst_of_bitstrings = make_permutations(num_digits, vals_per_dig)
        
    return {str_ : [] for str_ in lst_of_bitstrings}

def make_permutations(num_digits, vals_per_dig=2):
    """Make all permutations for a bit string of length num_digits
    and vals_per_dig values per digit. Hardcoded for work for binary vals per var"""
    if num_digits == 1:
        return [str(i) for i in range(vals_per_dig)]
    else:
        small_perms = make_permutations(num_digits - 1, vals_per_dig)
        # hardcoded for work for binary vals per var
        return ['0' + bit_str for bit_str in small_perms] + ['1' + bit_str for bit_str in small_perms]
    
def make_single_cond_prob_dct_of_lsts(num_variables=6, vals_per_var=2):
    """@Params
    num_variables = number of states that we care about
    
    @Returns
    dict of lists. Keys have the format: 'o_{i}={val}', where val is either '0' or '1'
    """
    res = {}
    for i in range(1, num_variables + 1):
        for val in range(vals_per_var):
            res[f'{i}{val}'] = []

    return res