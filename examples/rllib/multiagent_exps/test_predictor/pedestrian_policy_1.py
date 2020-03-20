from examples.rllib.multiagent_exps.bayesian_1_env import make_flow_params, setup_exps_PPO    
    
import inspect

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

from ray.tune import run_experiments

from flow.core.util import emission_to_csv
from flow.utils.registry import make_create_env
from flow.utils.rllib import get_rllib_config
from flow.utils.rllib import get_rllib_pkl

EXAMPLE_USAGE = """
example usage:
    python ./visualizer_rllib.py /ray_results/experiment_dir/result_dir 1

Here the arguments are:
1 - the path to the simulation results
2 - the number of the checkpoint    
"""

def test_predictor(args):
    flow_params = make_flow_params(render=True)
    alg_run, env_name, config = setup_exps_PPO(flow_params)    

    config_run = config['env_config']['run'] if 'run' in config['env_config'] \
        else None
    # Run on only one cpu for rendering purposes
    config['num_workers'] = 0

    if config_run:
        agent_cls = get_agent_class(config_run)
    
    ray.init(num_cpus=1)

    exp_tag = {
        'run': alg_run,
        'env': env_name,
        'checkpoint_freq': 25,
        "max_failures": 10,
        'stop': {
            'training_iteration': 100
        },
        'config': config,
        "num_samples": 1,
    }

    run_experiments(
        {
            flow_params["exp_tag"]: exp_tag
         },
    )



def create_parser():
    """Create the parser to capture CLI arguments."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='[Flow] Evaluates a reinforcement learning agent '
                    'given a checkpoint.',
        epilog=EXAMPLE_USAGE)
    parser.add_argument(
        '--num_rollouts',
        type=int,
        default=1,
        help='The number of rollouts to visualize.')
    parser.add_argument(
        '--horizon',
        type=int,
        help='Specifies the horizon.')
    return parser

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    test_predictor(args)