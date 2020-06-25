"""Get the agent to model the behaviour of vehicles"""
import argparse

import ray
try:
    from ray.rllib.agents.agent import get_agent_class
except ImportError:
    from ray.rllib.agents.registry import get_agent_class

from flow.controllers.imitation_learning.utils import *
from flow.controllers.imitation_learning.imitating_network import *
from flow.controllers.imitation_learning.utils_tensorflow import *
from flow.utils.registry import make_create_env

EXAMPLE_USAGE = """
example usage:
    python ./visualizer_rllib.py /ray_results/experiment_dir/result_dir 1
Here the arguments are:
1 - the path to the simulation results
2 - the number of the checkpoint
"""
def get_inference_network(path, flow_params):
    """Load and return imitation policy at path

    Parameters
    ----------
    path:
        path from which to load imitation policy from
    flow_params: dict
        dict to re-create the env

    Returns
    -------
    inference_network: tf_network object (is this what it's called?)
        Call the get_accel_gaussian_params_from_observation() method to get acceleration logits 
        resulting from an observation
    """
    # Look here for the params needed
    # class Args:
    #     def __init__(self):
    #         self.horizon = 400
    #         self.algo = "PPO"
    #         self.load_model=True
    #         self.load_path=path
    #         self.randomize_cars=True # might be wrong

    # args = Args()

    # might be necessary to add a load_model flag somewhere 
    create_env, _ = make_create_env(flow_params)
    env = create_env()
    obs_dim = env.observation_space.shape[0]
    action_dim = (1,)[0]

    sess = create_tf_session()
    inference_network = ImitatingNetwork(env, sess, action_dim, obs_dim, None, None, stochastic=True, load_model=True, load_path=path)

    return inference_network
    

# def create_parser():
#     """Create the parser to capture CLI arguments."""
#     parser = argparse.ArgumentParser(
#         formatter_class=argparse.RawDescriptionHelpFormatter,
#         description='[Flow] Evaluates a reinforcement learning agent '
#                     'given a checkpoint.',
#         epilog=EXAMPLE_USAGE)

#     # required input parameters
#     parser.add_argument(
#         'result_dir', type=str, help='Directory containing results')
#     parser.add_argument('checkpoint_num', type=str, help='Checkpoint number.')

#     # optional input parameters
#     parser.add_argument(
#         '--run',
#         type=str,
#         help='The algorithm or model to train. This may refer to '
#              'the name of a built-on algorithm (e.g. RLLib\'s DQN '
#              'or PPO), or a user-defined trainable function or '
#              'class registered in the tune registry. '
#              'Required for results trained with flow-0.2.0 and before.')
#     parser.add_argument(
#         '--num_rollouts',
#         type=int,
#         default=1,
#         help='The number of rollouts to visualize.')
#     parser.add_argument(
#         '--gen_emission',
#         action='store_true',
#         help='Specifies whether to generate an emission file from the '
#              'simulation')
#     parser.add_argument(
#         '--evaluate',
#         action='store_true',
#         help='Specifies whether to use the \'evaluate\' reward '
#              'for the environment.')
#     parser.add_argument(
#         '--render_mode',
#         type=str,
#         default='sumo_gui',
#         help='Pick the render mode. Options include sumo_web3d, '
#              'rgbd and sumo_gui')
#     parser.add_argument(
#         '--save_render',
#         action='store_true',
#         help='Saves a rendered video to a file. NOTE: Overrides render_mode '
#              'with pyglet rendering.')
#     parser.add_argument(
#         '--horizon',
#         type=int,
#         help='Specifies the horizon.')
    
#     parser.add_argument('--grid_search', action='store_true', default=False,
#                         help='If true, a grid search is run')
#     parser.add_argument('--run_mode', type=str, default='local',
#                         help="Experiment run mode (local | cluster)")
#     parser.add_argument('--algo', type=str, default='TD3',
#                         help="RL method to use (PPO, TD3, MADDPG)")
#     parser.add_argument("--pedestrians",
#                         help="use pedestrians, sidewalks, and crossings in the simulation",
#                         action="store_true")
    
#     return parser
