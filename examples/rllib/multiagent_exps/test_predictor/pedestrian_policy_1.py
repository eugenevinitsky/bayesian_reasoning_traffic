"""Visualizer for rllib experiments.
Attributes
----------
EXAMPLE_USAGE : str
    Example call to the function, which is
    ::
        python ./visualizer_rllib.py /tmp/ray/result_dir 1
parser : ArgumentParser
    Command-line argument parser
"""

import argparse
from datetime import datetime
import gym
import numpy as np
import os
import sys
import time

import ray
try:
    from ray.rllib.agents.agent import get_agent_class
except ImportError:
    from ray.rllib.agents.registry import get_agent_class
from ray.tune.registry import register_env

from flow.core.util import emission_to_csv
from flow.utils.registry import make_create_env
from flow.utils.rllib import get_flow_params
from flow.utils.rllib import get_rllib_config
from flow.utils.rllib import get_rllib_pkl


EXAMPLE_USAGE = """
example usage:
    python ./visualizer_rllib.py /ray_results/experiment_dir/result_dir 1
Here the arguments are:
1 - the path to the simulation results
2 - the number of the checkpoint
"""

def run_env(env, agent, config, flow_params):

    if config.get('multiagent', {}).get('policies', None):
        multiagent = True
        rets = {}
        # map the agent id to its policy
        policy_map_fn = config['multiagent']['policy_mapping_fn'].func
        for key in config['multiagent']['policies'].keys():
            rets[key] = []
    else:
        multiagent = False
        rets = []

    if config['model']['use_lstm']:
        use_lstm = True
        if multiagent:
            state_init = {}
            # map the agent id to its policy
            policy_map_fn = config['multiagent']['policy_mapping_fn'].func
            size = config['model']['lstm_cell_size']
            for key in config['multiagent']['policies'].keys():
                state_init[key] = [np.zeros(size, np.float32),
                                   np.zeros(size, np.float32)]
        else:
            state_init = [
                np.zeros(config['model']['lstm_cell_size'], np.float32),
                np.zeros(config['model']['lstm_cell_size'], np.float32)
            ]
    else:
        use_lstm = False

    env.restart_simulation(
        sim_params=flow_params['sim'], render=flow_params['sim'].render)

    # Simulate and collect metrics
    final_outflows = []
    final_inflows = []
    mean_speed = []
    std_speed = []
    num_pedestrian_crash = 0

    # HARDCODED variable names
    prob_action_given_ped = []
    prob_action_given_no_ped = []

    # updated priors list
    probs_ped_given_action = []
    probs_no_ped_given_action = []

    # fixed priors list
    probs_ped_given_action_fixed_priors = []
    probs_no_ped_given_action_fixed_priors = []

    # update these priors        
    prior_prob_ped = 0.5
    prior_prob_no_ped = 0.5

    # fixed prior prob
    fixed_prior_prob_ped = 0.5
    fixed_prior_prob_no_ped = 0.5

    visible_pedestrian = []

    for i in range(args.num_rollouts + 400):
        vel = []
        state = env.reset()
        if multiagent:
            ret = {key: [0] for key in rets.keys()}
        else:
            ret = 0
        for _ in range(flow_params['env'].horizon + 400):
            vehicles = env.unwrapped.k.vehicle
            pedestrian = env.unwrapped.k.pedestrian
            vel.append(np.mean(vehicles.get_speed(vehicles.get_ids())))
            if multiagent:
                action = {}
                logits = {}
                for agent_id in state.keys():
                    if use_lstm:
                        action[agent_id], state_init[agent_id], logits = \
                            agent.compute_action(
                            state[agent_id], state=state_init[agent_id],
                            policy_id=policy_map_fn(agent_id))
                    else:
                        # import ipdb;ipdb.set_trace()
                        # TODO(KL) HARD CODED state alert! is_ped_visible is the 5th item in the state vector (i.e. index-4)
                        ped_idx = 4
                        curr_ped = state[agent_id][ped_idx]
                        visible_pedestrian.append(curr_ped)
                        # no
                        flipped_ped = 1 if curr_ped == 0 else 0
                        ped_flipped_state = np.copy(state[agent_id])
                        ped_flipped_state[ped_idx] = flipped_ped

                        action[agent_id], _, logit_actual = agent.compute_action(
                            state[agent_id], policy_id=policy_map_fn(agent_id), full_fetch=True)
                            
                        _, _, logit_flipped = agent.compute_action(
                            ped_flipped_state, policy_id=policy_map_fn(agent_id), full_fetch=True)

                        actual_mu, actual_ln_sigma = logit_actual['behaviour_logits']
                        flipped_mu, flipped_ln_sigma = logit_flipped['behaviour_logits']

                        actual_sigma = np.exp(actual_ln_sigma)
                        flipped_sigma = np.exp(flipped_ln_sigma)

                        actual_action = action[agent_id][0]
                        print('Prob of action from when there is a pedestrian vs prob from of action when there is no pedestrian')
                        # print(accel_pdf(actual_mu, actual_sigma, actual_action))
                        # print(accel_pdf(flipped_mu, flipped_sigma, actual_action))
                        # import ipdb; ipdb.set_trace()

                        # actual mu and actual sigma are the mu/sigma values arising from assuming there is a pedestrian
                        unnormed_prob_action_given_ped = accel_pdf(actual_mu, actual_sigma, actual_action)
                        unnormed_prob_action_given_no_ped = accel_pdf(flipped_mu, flipped_sigma, actual_action)

                        pr_a_given_ped = unnormed_prob_action_given_ped / (unnormed_prob_action_given_ped + unnormed_prob_action_given_no_ped)
                        pr_a_given_no_ped = 1 - pr_a_given_ped

                        prob_action_given_ped.append(pr_a_given_ped)
                        prob_action_given_no_ped.append(pr_a_given_no_ped)

                        # updating priors Pr(ped | action)
                        prob_ped_given_action = (pr_a_given_ped * prior_prob_ped) / ((pr_a_given_ped * prior_prob_ped)  + (pr_a_given_no_ped * prior_prob_no_ped))
                        prob_no_ped_given_action = (pr_a_given_no_ped * prior_prob_no_ped) / ((pr_a_given_ped * prior_prob_ped)  + (pr_a_given_no_ped * prior_prob_no_ped))
                        
                        prior_prob_ped = prob_ped_given_action
                        prior_prob_no_ped = prob_no_ped_given_action

                        probs_ped_given_action.append(prob_ped_given_action)
                        probs_no_ped_given_action.append(prob_no_ped_given_action)

                        # fixed priors Pr(ped | action)
                        prob_ped_given_action_fixed_prior = (pr_a_given_ped * 0.5) / ((pr_a_given_ped * 0.5)  + (pr_a_given_no_ped * 0.5))
                        prob_no_ped_given_action_fixed_prior = (pr_a_given_no_ped * 0.5) / ((pr_a_given_ped * 0.5)  + (pr_a_given_no_ped * 0.5))
                        
                        probs_ped_given_action_fixed_priors.append(prob_ped_given_action_fixed_prior)
                        probs_no_ped_given_action_fixed_priors.append(prob_no_ped_given_action_fixed_prior)

            else:
                action = agent.compute_action(state)
            state, reward, done, _ = env.step(action)
            if multiagent:
                for actor, rew in reward.items():
                    ret[policy_map_fn(actor)][0] += rew
            else:
                ret += reward
            if multiagent and done['__all__']:
                break
            if not multiagent and done:
                break

            for rl_id in vehicles.get_rl_ids():
                num_collision = len(vehicles.get_pedestrian_crash(rl_id, pedestrian))
                num_pedestrian_crash += num_collision
        
        # plot_2_licnes(prob_action_given_ped, prob_action_given_no_ped)
        plot_2_lines(probs_ped_given_action, probs_no_ped_given_action, updated_priors=True, viewable_ped=visible_pedestrian)
        plot_2_lines(probs_ped_given_action_fixed_priors, probs_no_ped_given_action_fixed_priors, updated_priors=False, viewable_ped=visible_pedestrian)


def create_env(args, flow_params):
    # Create and register a gym+rllib env
    create_env, env_name = make_create_env(params=flow_params, version=0)
    register_env(env_name, create_env)

    # check if the environment is a single or multiagent environment, and
    # get the right address accordingly
    # single_agent_envs = [env for env in dir(flow.envs)
    #                      if not env.startswith('__')]

    # if flow_params['env_name'] in single_agent_envs:
    #     env_loc = 'flow.envs'
    # else:
    #     env_loc = 'flow.envs.multiagent'

    # Start the environment with the gui turned on and a path for the
    # emission file
    env_params = flow_params['env']
    env_params.restart_instance = False
    if args.evaluate:
        env_params.evaluate = True

    # lower the horizon if testing
    if args.horizon:
        env_params.horizon = args.horizon

    sim_params = flow_params['sim']
    # pick your rendering mode
    if args.render_mode == 'sumo_web3d':
        sim_params.num_clients = 2
        sim_params.render = False
    elif args.render_mode == 'drgb':
        sim_params.render = 'drgb'
        sim_params.pxpm = 4
    elif args.render_mode == 'sumo_gui':
        sim_params.render = True
        print('NOTE: With render mode {}, an extra instance of the SUMO GUI '
              'will display before the GUI for visualizing the result. Click '
              'the green Play arrow to continue.'.format(args.render_mode))
    elif args.render_mode == 'no_render':
        sim_params.render = False
    if args.save_render:
        sim_params.render = 'drgb'
        sim_params.pxpm = 4
        sim_params.save_render = True

    env = create_env()
    return env, env_name


def create_agent(args, flow_params):
    """Visualizer for RLlib experiments.
    This function takes args (see function create_parser below for
    more detailed information on what information can be fed to this
    visualizer), and renders the experiment associated with it.
    """
    result_dir = args.result_dir if args.result_dir[-1] != '/' \
        else args.result_dir[:-1]

    config = get_rllib_config(result_dir)

    # check if we have a multiagent environment but in a
    # backwards compatible way
    if config.get('multiagent', {}).get('policies', None):
        pkl = get_rllib_pkl(result_dir)
        config['multiagent'] = pkl['multiagent']

    # Run on only one cpu for rendering purposes
    config['num_workers'] = 0

    # Determine agent and checkpoint
    config_run = config['env_config']['run'] if 'run' in config['env_config'] \
        else None
    if args.run and config_run:
        if args.run != config_run:
            print('visualizer_rllib.py: error: run argument '
                  + '\'{}\' passed in '.format(args.run)
                  + 'differs from the one stored in params.json '
                  + '\'{}\''.format(config_run))
            sys.exit(1)

    if args.run:
        agent_cls = get_agent_class(args.run)
    elif config_run:
        agent_cls = get_agent_class(config_run)


    else:
        print('visualizer_rllib.py: error: could not find flow parameter '
              '\'run\' in params.json, '
              'add argument --run to provide the algorithm or model used '
              'to train the results\n e.g. '
              'python ./visualizer_rllib.py /tmp/ray/result_dir 1 --run PPO')
        sys.exit(1)

    # TODO(@evinitsky) duplication
    env, env_name = create_env(args, flow_params)

    # create the agent that will be used to compute the actions
    agent = agent_cls(env=env_name, config=config)
    checkpoint = result_dir + '/checkpoint_' + args.checkpoint_num
    checkpoint = checkpoint + '/checkpoint-' + args.checkpoint_num
    agent.restore(checkpoint)

    return agent, config

def accel_pdf(mu, sigma, actual):
    """Return pdf evaluated at actual acceleration"""
    coeff = 1 / np.sqrt(2 * np.pi * (sigma**2))
    exp = -0.5 * ((actual - mu) / sigma)**2
    return coeff * np.exp(exp)

def run_transfer(args):
    # run transfer on the bayesian 1 env first
    from examples.rllib.multiagent_exps.bayesian_0_no_grid_env import make_flow_params as bayesian_1_flow_params
    bayesian_1_params = bayesian_1_flow_params(pedestrians=True, render=True)
    env, env_name = create_env(args, bayesian_1_params)
    agent, config = create_agent(args, flow_params=bayesian_1_params)
    run_env(env, agent, config, bayesian_1_params)

def plot_2_lines(actual_pdfs, flipped_pdfs, updated_priors=True, viewable_ped=False):
    import matplotlib.pyplot as plt
    x = np.arange(len(actual_pdfs))
    plt.plot(x, actual_pdfs)
    plt.plot(x, flipped_pdfs)
    if viewable_ped:
        plt.plot(x, viewable_ped)
    if updated_priors:
        plt.legend(['Pr(ped | action) using updated priors', 'Pr(no_ped | action) using updated priors'], loc='upper left')
    else:
        plt.legend(['Pr(ped | action) using fixed priors of Pr(ped) = 0.5', 'Pr(no_ped | action) using fixed priors of Pr(ped) = 0.5'], loc='upper left')
    plt.show()


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
    return parser

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    ray.init(num_cpus=1)
    run_transfer(args)