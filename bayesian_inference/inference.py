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

pp = 14516 / (107 + 14516)
pno = 107 / (107 + 164579)
nop = 107 / (107 + 14516)
nono = 164579 / (164579 + 107)

a = np.array([nono, pno])
b = np.array([nop, pp])

TRANSITION_MATRIX = np.array([a, b])

PED_IDX_LST = [10, 11, 12, 13]
PED_FRONT = PED_IDX_LST[0]
PED_BACK = PED_IDX_LST[-1]

def get_filtered_posteriors(action, dummy_obs, priors, agent, filter_matrix, policy_type="Imitation", num_locs=4):
    """Black box predictor of the probability of pedestrian in each of the 4 crosswalk locations
    
    Parameters
    ----------
    action: float
        actions taken by the vehicle is equivalent to the acceleration in m / s^2
    dummy_obs: [non-ped-self-obs] + [None] * num_locs + [other-veh-obs] * (num_max_obj - 1)
        self_obs: [yaw, speed, turn_num, edge_pos]
        other_veh_obs: ["rel_x", "rel_y", "speed", "yaw", "arrive_before"]
        [None] * num_locs: dummy place holders for the 4 ped params
    priors: dict[str:float]
        (single pedestrian location observation : probability)
        ex: priors["o_3 = 1"] = 0.1

        (total of 2^4 pedestrian observation combinations and their 
        corresponding updated prior probabilities)

    agent: agent_object
        ray.rllib agent / policy object used to model the actions of vehicles  
        TODO(KL) Terminology - agent or policy? RL land :D
    policy_type: string
        type of policy used to model vehicle behavior and do inference
    
    Returns
    -------
    prob_ped_in_cross_walk: [prob_ped_in_cw_0, ..., prob_ped_in_cw_3]
        posterior probabilties of peds being on certain crosswalks
        computed using:
        - a selected policy
        - Bayes' rule
        - the given prior probabilities
        - a filter matrix, F
    single_posteriors: dict[str:float]
        (pedestrian observation combination : probability) to be used as priors for the next inference round
        ex: priors["o_3 = 1"] = 0.1

        (total of 2^4 pedestrian observation combinations and their 
        corresponding updated prior probabilities)    
    """
    joint_priors = {}
    joint_posteriors = {}
    prob_ped_in_cross_walk = [None]*4

    flag_set = ("0", "1")
    single_ped_combs_str = single_ped_posteriors_strs(num_locs, flag_set)
    joint_ped_combos_str = all_ped_combos_strs(num_locs, flag_set)

    
    # p(o_1 = 1) = 0.887, p(o_2 = 1) = 0.443
    if priors == {}:
        priors = {comb : [1 / len(flag_set)] for comb in single_ped_combs_str}
    # p(o_0, o_1, o_2, o_3)

    # 3 Update joint priors p(o|a) = \prod_{o_i \in o} p(o_i)
    for str_comb in joint_ped_combos_str:
        joint_prior = 1
        
        single_ped_lst = joint_ped_combo_str_to_single_ped_combo(str_comb)
        for single_ped in single_ped_lst:
            joint_prior *= priors[single_ped][-1]

        joint_priors[str_comb] = joint_prior

    # 2 
    C = 0
    for str_comb, lst_comb in zip(joint_ped_combos_str, joint_ped_combos_int_list):

        s_all_modified = np.copy(dummy_obs) # s_all_modified = hypothetical state that an agent observes
        s_all_modified[ped_front : ped_back + 1] = lst_comb

        if policy_type == "DQN":
            _, _, logit = agent.compute_action(s_all_modified, policy_id='av', full_fetch=True)
            q_vals = logit['q_values']
            soft_max = scipy.special.softmax(q_vals)
            joint_likelihood_density = soft_max[action_index]

        elif policy_type == "imitation":
            mu, sigma = agent.get_accel_gaussian_params_from_observation(s_all_modified)
            sigma = sigma[0][0]
            mu = mu[0]

            # f(a|e)
            joint_likelihood_density = accel_pdf(mu, sigma, action)

        joint_likelihood_densities[str_comb].append(joint_likelihood_density)

        # Get p(e), for f(a|e) p(e)
        updated_prior = joint_priors[str_comb]

        C += joint_likelihood_density * updated_prior
    
    # 5 p(e|a) joint posterior masses
    for str_comb in joint_ped_combos_str:
        # f(a|e)
        joint_likelihood_density = joint_likelihood_densities[str_comb][-1]
        # p(e)
        joint_prior = joint_priors_updated[str_comb][-1]

        # p(e|a) = f(a|e) p(e) / C
        joint_posterior = joint_likelihood_density * updated_prior / C
        joint_posteriors[str_comb] = joint_posterior

    # 6 Single posterior probabilities p(o_i = 1|a) = sum relevant posteriors
    for loc_ in range(num_locs):
        for val_ in flag_set:
            single_posterior = 0
            for key in ped_combos_one_loc_fixed_strs(loc_, val_):
                single_posterior += joint_posteriors[key]
            single_posterior_str = f'o_{loc_} = {val_}'
            single_posteriors[single_posterior_str] = single_posterior

    for loc_ in range(num_locs):
        val0, val1 = "0", "1"
        single_0 = f'o_{loc_} = {val0}'
        single_1 = f'o_{loc_} = {val1}'
        filtered = TRANSITION_MATRIX @ np.array([single_posteriors_filter[single_0][-1], single_posteriors_filter[single_1][-1]])
        single_posteriors[single_0], single_posteriors_filter[single_1] = filtered[0], filtered[1]

    for loc__ in range(num_locs):
        single_1 = f'o_{loc__} = 1'
        posterior_prob_ped[loc_] = single_posteriors[single_1]

    return prob_ped_in_cross_walk, single_posteriors

    #############################
    #############################
    ###   Utility functions   ###
    #############################
    #############################

def joint_ped_combo_str_to_single_ped_combo(joint_ped_combo_str):
    """Given a string of format '0 1 0 -1', return a list of all relevant single ped strings
    i.e ['o_0 = 0', 'o_1 = 1', 'o_2 = 0', 'o_3 = -1']
    """
    res = []
    jnt_ped_list = joint_ped_combo_str.split(" ")
    for loc, val in enumerate(jnt_ped_list):
        res.append(single_posterior_to_str(loc, val))
    return res

def single_posterior_to_str(loc, val):
    return f'o_{loc} = {val}'

def all_ped_combos_strs(num_locs=4, val_set=("0", "1")):
    """Return a list of all pedestrian observation combinations (in string format) for a vehicle under the 4 location scheme"""
    res = []
    lsts = all_ped_combos_lsts(num_locs, val_set)
    for lst in lsts:
        res.append(" ".join(lst))
    return res

def all_ped_combos_lsts(num_locs=4, val_set=("0", "1")):
    """Return a list of all pedestrian observation combinations (in list format) for a vehicle under the 4 location scheme"""
    res = []
    if num_locs == 0:
        return []
    if num_locs == 1:
        return [[flag] for flag in val_set]

    for comb in all_ped_combos_lsts(num_locs - 1, val_set):
        # append a flag for all possible flags
        for flag in val_set:
            appended = comb + [flag]
            res.append(appended)
            
    return res

def ped_combos_one_loc_fixed_strs(fixed_loc, fixed_val, num_locs=4, val_set=("0", "1")):
    """Return a list of all ped observation combs for a vehicle under the 4 location scheme
    SUBJECT TO fixed_loc == fix_val
    
    This is handy for summation selection in equation (4) of the derivation
    
    @Parameters
    fixed_loc: int
        location from 0, 1, 2, 3
    fixed_val: int
        location from -1, 0, 1
    """    
    res = []
    lsts = ped_combos_one_loc_fixed_lsts(fixed_loc, fixed_val, num_locs, val_set)
    for lst in lsts:
        res.append(" ".join(lst))
    return res

def ped_combos_one_loc_fixed_lsts(fixed_loc, fixed_val, num_locs=4, val_set=("0", "1")):
    """Return a list of all ped observation combs for a vehicle under the 4 location scheme
    SUBJECT TO fixed_loc == fix_val
    
    This is handy for summation selection in equation (4) of the derivation
    
    @Parameters
    fixed_loc: int
        location from 0, 1, 2, 3
    fixed_val: int
        location from -1, 0, 1
    """
    fixed_val = str(fixed_val)
    assert fixed_loc < num_locs and (fixed_val in val_set or str(fixed_val) in val_set)
    
    res = []
    for comb in all_ped_combos_lsts(num_locs - 1, val_set):
        # insert fixed val at correct position
        left = comb[:fixed_loc]
        right = comb[fixed_loc:]
        res.append(left + [fixed_val] + right)
    
    return res

def single_cond_prob_to_str(grid_idx, val, num_indices = 6):
    """Generate the string representing the probability:
    
    Pr(o_i = val)
    
    ex:
    For Pr(o_2 = 1), we'd have the string '21'
    NB we're 1-indexing here
    """
    assert grid_idx >= 1 and grid_idx <= num_indices
    return str(grid_idx) + str(val)

# better name for this? 
def ped_combos_for_single_cond_prob(grid_idx, val, output_len=6):
    """Helper function for computing a 'single' conditional probability e.g. p(o_3 = 1 | action)
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

def initial_prior_probs(num_digits=4, vals_per_dig=2):
    """Returns a dict with values of all permutations of bitstrings of length num_digits. 
    Each digit can take a value from 0 to (vals_per_dig - 1)"""
    uniform_prob = 1 / (vals_per_dig ** num_digits)
    res = make_dct_of_lsts(num_digits, vals_per_dig)
    for key in res.keys():
        res[key] = res[key] + [uniform_prob]
    return res

def make_dct_of_lsts(num_digits=4, vals_per_dig=2):
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
    
def single_ped_posteriors_strs(num_variables=4, val_set=("0", "1")):
    """
    @Params
    num_variables = number of ped locations
    
    @Returns
    list of strings. Strings have the format: 'o_{i}={val}', where val is in val_set
    """
    res = []
    for i in range(num_variables):
        for flag in val_set:
            res.append(f'o_{i} = {flag}')
    return res

def accel_pdf(mu, sigma, actual):
    """Return pdf evaluated at actual acceleration"""
    coeff = 1 / np.sqrt(2 * np.pi * (sigma**2))
    exp = -0.5 * ((actual - mu) / sigma)**2
    return coeff * np.exp(exp)

def run_transfer(args, action_pairs_lst=[], imitation_model=True, is_discrete=False):
    """To run inference using the imitation model, set imitation_model True, is_discrete False.
    To run DQN inference, set imitation_model = False, is_discrete = True"""
    # run transfer on the bayesian 1 env first
    action_pairs_lst = []
    bayesian_0_params = bayesian_1_flow_params(args, pedestrians=True, render=False)
    env, env_name = create_env(args, bayesian_0_params)
    if imitation_model:
        agent = get_inference_network(os.path.abspath("../flow/controllers/imitation_learning/model_files/c_test.h5"), flow_params=bayesian_0_params)
    else:
        agent, config = create_agent(args, flow_params=bayesian_0_params)
    action_pairs_lst = run_env(env, agent, bayesian_0_params, is_discrete)
    
    return action_pairs_lst
def plot_2_lines(y1, y2, legend, viewable_ped=False):
    x = np.arange(len(y1))
    plt.plot(x, y1)
    plt.plot(x, y2)
    if viewable_ped:
        plt.plot(x, viewable_ped)
    plt.legend(legend, bbox_to_anchor=(0.5, 1.05), loc=3, borderaxespad=0.)
    plt.ylim(-0.1, 1.1)

    plt.draw()
    plt.pause(0.001)
    
def plot_lines(y_val_lsts, legends):
    assert len(y_val_lsts) == len(legends)
    x = np.arange(len(y_val_lsts[0]))
    for y_vals in y_val_lsts:
        plt.plot(x, y_vals)
    plt.ylim(-0.1, 1.1)

    plt.legend(legends, bbox_to_anchor=(0.5, 1.05), loc=3, borderaxespad=0.)
    plt.draw()
    plt.pause(0.001)
    
def create_parser():
    """Create the parser to capture CLI arguments."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='[Flow] Evaluates a reinforcement learning agent '
                    'given a checkpoint.',
        epilog=EXAMPLE_USAGE)

    # required input parameters
    parser.add_argument(
        'result_dir', type=str, help='Directory containing results')
    parser.add_argument('checkpoint_num', type=str, help='Checkpoint number.')

    # optional input parameters
    parser.add_argument(
        '--run',
        type=str,
        help='The algorithm or model to train. This may refer to '
             'the name of a built-on algorithm (e.g. RLLib\'s DQN '
             'or PPO), or a user-defined trainable function or '
             'class registered in the tune registry. '
             'Required for results trained with flow-0.2.0 and before.')
    parser.add_argument(
        '--num_rollouts',
        type=int,
        default=1,
        help='The number of rollouts to visualize.')
    parser.add_argument(
        '--gen_emission',
        action='store_true',
        help='Specifies whether to generate an emission file from the '
             'simulation')
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='Specifies whether to use the \'evaluate\' reward '
             'for the environment.')
    parser.add_argument(
        '--render_mode',
        type=str,
        default='sumo_gui',
        help='Pick the render mode. Options include sumo_web3d, '
             'rgbd and sumo_gui')
    parser.add_argument(
        '--save_render',
        action='store_true',
        help='Saves a rendered video to a file. NOTE: Overrides render_mode '
             'with pyglet rendering.')
    parser.add_argument(
        '--horizon',
        type=int,
        help='Specifies the horizon.')
    parser.add_argument(
        "--randomize_vehicles", default=True,
        help="randomize the number of vehicles in the system and where they come from",
        action="store_true")
    
    parser.add_argument('--grid_search', action='store_true', default=False,
                        help='If true, a grid search is run')
    parser.add_argument('--run_mode', type=str, default='local',
                        help="Experiment run mode (local | cluster)")
    parser.add_argument('--algo', type=str, default='TD3',
                        help="RL method to use (PPO, TD3, MADDPG)")
    parser.add_argument("--pedestrians",
                        help="use pedestrians, sidewalks, and crossings in the simulation",
                        action="store_true")
    
    return parser