import time
import numpy as np
import gym
import os
from flow.utils.registry import make_create_env
from examples.rllib.multiagent_exps.bayesian_0_no_grid_env import make_flow_params as bay_0_make_flow_params
from utils import *
from imitating_network import *
from utils_tensorflow import *
from flow.core.experiment import Experiment
from flow.core.params import SimParams


# script for bay0_no_grid flow_params
class Args:
    def __init__(self):
        self.horizon = 400
        self.algo = "PPO"
args = Args()
bay_0_make_flow_params
flow_params = bay_0_make_flow_params(args, pedestrians=True, render=True)

def run_experiment():
    create_env, _ = make_create_env(flow_params)
    env = create_env()

    obs_dim = env.observation_space.shape[0]
    action_dim = (1,)[0]

    sess = create_tf_session()
    action_network = ImitatingNetwork(sess, action_dim, obs_dim, None, None, None, None, load_model=True, load_path='flow/controllers/imitation_learning/model_files/bay0')

    def get_rl_actions(state):
        rl_actions = {}
        for vehicle_id in state.keys():
            obs = state[vehicle_id]
            action = action_network.get_accel_from_observation(obs)
            rl_actions[vehicle_id] = action
        return rl_actions

    exp = Experiment(flow_params)
    exp.run(num_runs=1, rl_actions=get_rl_actions, convert_to_csv=True)



def run_rollout():

    create_env, _ = make_create_env(flow_params)
    env = create_env()

    obs_dim = env.observation_space.shape[0]
    action_dim = (1,)[0]

    sess = create_tf_session()
    action_network = ImitatingNetwork(sess, action_dim, obs_dim, None, None, None, None, load_existing=True, load_path='flow/controllers/imitation_learning/model_files/bay0')

    init_state = env.reset()

    test_state = np.array([[1.0,1.0,1.0]], dtype='float32')

    reward = 0
    while(True):
        rl_vehicles = env.k.vehicle.get_rl_ids()
        if len(rl_vehicles) == 0:
            observation_dict, reward_dict, done_dict, _ = env.step(None)
            reward += sum(reward_dict.values())
            if done_dict['__all__']:
                break
            continue

        rl_actions = {}
        observations = env.get_state()

        for vehicle_id in rl_vehicles:
            obs = observations[vehicle_id]
            action = action_network.get_accel_from_observation(obs)
            rl_actions[vehicle_id] = action


        observation_dict, reward_dict, done_dict, _ = env.step(rl_actions)
        reward += sum(reward_dict.values())
        if done_dict['__all__']:
            break

    print("Final Reward: ", reward)

if __name__ == "__main__":
    run_experiment()
