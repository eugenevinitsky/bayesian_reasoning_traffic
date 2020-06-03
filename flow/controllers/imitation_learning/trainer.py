import time
import pickle
import numpy as np
import gym
import os
import argparse

from collections import OrderedDict

from flow.utils.registry import make_create_env
from imitating_controller import ImitatingController
from imitating_network import ImitatingNetwork
from flow.controllers.car_following_models import IDMController
from flow.controllers.velocity_controllers import FollowerStopper
from flow.core.params import SumoCarFollowingParams
import tensorflow as tf
from utils import *
from utils_tensorflow import *


class Trainer(object):
    """
    Class to initialize and run training for imitation learning (with DAgger)
    """

    def __init__(self, params, submodule, render=False):
        """
        Parameters
        __________
        params: dict
            Dictionary of parameters used to run imitation learning
        submodule: Module
            Python module for file containing flow_params
        """
        class Args:
            def __init__(self):
                self.horizon = 400
                self.algo = "PPO"

        args = Args()

        # get flow params
        self.flow_params = submodule.make_flow_params(args, pedestrians=True, render=render)

        # setup parameters for training
        self.params = params
        self.sess = create_tf_session()

        # environment setup
        create_env, _ = make_create_env(self.flow_params)
        self.env = create_env()

        # vehicle setup
        self.multiagent = self.params['multiagent'] # multiagent or singleagent env

        if not self.multiagent and self.env.action_space.shape[0] > 1:
            # use sorted rl ids if the method exists (e.g.. singlagent straightroad)
            try:
                self.vehicle_ids = self.env.get_sorted_rl_ids()
            except:
                self.vehicle_ids = self.k.vehicle.get_rl_ids()
        else:
            # use get_rl_ids if sorted_rl_ids doesn't exist
            self.vehicle_ids = self.env.k.vehicle.get_rl_ids()

        # neural net setup
        obs_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]

        self.params['action_dim'] = action_dim
        self.params['obs_dim'] = obs_dim

        # initialize neural network class and tf variables
        self.action_network = ImitatingNetwork(self.sess, self.params['action_dim'], self.params['obs_dim'], self.params['fcnet_hiddens'], self.params['replay_buffer_size'], stochastic=self.params['stochastic'], variance_regularizer=self.params['variance_regularizer'])

        # tf.global_variables_initializer().run(session=self.sess)

        # controllers setup
        car_following_params = SumoCarFollowingParams()
        self.controllers = dict()

        # initialize controllers: save in a dictionary to avoid re-initializing a controller for a vehicle
        for vehicle_id in self.vehicle_ids:
            expert = IDMController(vehicle_id, car_following_params=car_following_params)
            imitator = ImitatingController(vehicle_id, self.action_network, self.multiagent, car_following_params=car_following_params)
            self.controllers[vehicle_id] = (imitator, expert)


    def run_training_loop(self, n_iter):
        """
        Trains imitator for n_iter iterations (each iteration collects new trajectories to put in replay buffer)

        Parameters
        __________
        n_iter :
            intnumber of iterations to execute training
        """

        # init vars at beginning of training
        # number of environment steps taken throughout training
        self.total_envsteps = 0

        for itr in range(n_iter):
            print("\n\n********** Iteration %i ************"%itr)
            # collect trajectories, to be used for training
            if itr == 0:
                # first iteration is behavioral cloning
                training_returns = self.collect_training_trajectories(itr, self.params['init_batch_size'])
            else:
                # other iterations use DAgger (trajectories collected by running imitator policy)
                training_returns = self.collect_training_trajectories(itr, self.params['batch_size'])

            paths, envsteps_this_batch = training_returns
            self.total_envsteps += envsteps_this_batch

            # add collected data to replay buffer in neural network class
            self.action_network.add_to_replay_buffer(paths)

            # train controller
            self.train_controller()

            if itr % 20 == 0:
                self.evaluate_controller(10)

    def collect_training_trajectories(self, itr, batch_size):
        """
        Collect (state, action, reward, next_state, terminal) tuples for training

        Parameters
        __________
        itr: int
            iteration of training during which function is called. Used to determine whether to run behavioral cloning or DAgger
        batch_size: int
            number of tuples to collect
        Returns
        _______
        paths: list
            list of trajectories
        envsteps_this_batch: int
            the sum over the numbers of environment steps in paths (total number of env transitions in trajectories collected)
        """
        print("\nCollecting data to be used for training...")
        max_decel = self.flow_params['env'].additional_params['max_decel']
        trajectories, envsteps_this_batch = sample_trajectories(self.env, self.controllers, self.action_network, batch_size, self.params['ep_len'], self.multiagent, use_expert= itr<self.params['n_bc_iter'], max_decel=max_decel)
        print(itr<self.params['n_bc_iter'])
        return trajectories, envsteps_this_batch

    def train_controller(self):
        """
        Trains controller for specified number of steps, using data sampled from replay buffer; each step involves running optimizer (i.e. Adam) once
        """

        print("Training controller using sampled data from replay buffer...")
        for train_step in range(self.params['num_agent_train_steps_per_iter']):
            # sample data from replay buffer
            ob_batch, ac_batch, expert_ac_batch = self.action_network.sample_data(self.params['train_batch_size'])
            # train network on sampled data
            self.action_network.train(ob_batch, expert_ac_batch)

    def evaluate_controller(self, num_trajs = 10):
        """
        Evaluates a trained imitation controller on similarity with expert with respect to action taken and total reward per rollout.

        Parameters
        __________
        num_trajs: int
            number of trajectories to evaluate performance on
        """

        print("\n\n********** Evaluation ************ \n")


        # collect imitator driven trajectories (along with corresponding expert actions)
        trajectories = sample_n_trajectories(self.env, self.controllers, self.action_network, num_trajs, self.params['ep_len'], self.multiagent, False)

        # initialize metrics
        total_imitator_steps = 0  # total number of environment steps taken across the n trajectories
        average_imitator_reward_per_rollout = 0 # average reward per rollout achieved by imitator

        action_errors = np.array([]) # difference in action (acceleration) taken between expert and imitator
        average_action_expert = 0 # average action taken, across all timesteps, by expert (used to compute % average)
        average_action_imitator = 0 # average action taken, across all timesteps, by imitator (used to compute % average)

        # compare actions taken in each step of trajectories (trajectories are controlled by imitator)
        for traj_tuple in trajectories:
            traj = traj_tuple[0]
            traj_len = traj_tuple[1]

            imitator_actions = traj['actions']
            expert_actions = traj['expert_actions']

            average_action_expert += np.sum(expert_actions)
            average_action_imitator += np.sum(imitator_actions)

            # use RMSE as action error metric
            action_error = (np.linalg.norm(imitator_actions - expert_actions)) / len(imitator_actions)
            action_errors = np.append(action_errors, action_error)

            total_imitator_steps += traj_len
            average_imitator_reward_per_rollout += np.sum(traj['rewards'])
    
        if len(trajectories) == 0:
            import ipdb; ipdb.set_trace()
            trajectories = sample_n_trajectories(self.env, self.controllers, self.action_network, num_trajs, self.params['ep_len'], self.multiagent, False)

        # compute averages for metrics
        average_imitator_reward_per_rollout = average_imitator_reward_per_rollout / len(trajectories)
        
        average_action_expert = average_action_expert / total_imitator_steps

        # collect expert driven trajectories (these trajectories are only used to compare average reward per rollout)
        expert_trajectories = sample_n_trajectories(self.env, self.controllers, self.action_network, num_trajs, self.params['ep_len'], self.multiagent, True)

        # initialize metrics
        total_expert_steps = 0
        average_expert_reward_per_rollout = 0

        # compare reward accumulated in trajectories collected via expert vs. via imitator
        for traj_tuple in expert_trajectories:
            traj = traj_tuple[0]
            traj_len = traj_tuple[1]
            total_expert_steps += traj_len
            average_expert_reward_per_rollout += np.sum(traj['rewards'])

        average_expert_reward_per_rollout = average_expert_reward_per_rollout / len(expert_trajectories)

        # compute percent errors (using expert values as 'ground truth')
        percent_error_average_reward = (np.abs(average_expert_reward_per_rollout - average_imitator_reward_per_rollout) / average_expert_reward_per_rollout) * 100

        percent_error_average_action = (np.abs(np.mean(action_errors)) / np.abs(average_action_expert)) * 100

        # Print results
        print("\nAverage reward per rollout, expert: ", average_expert_reward_per_rollout)
        print("Average reward per rollout, imitator: ", average_imitator_reward_per_rollout)
        print("% Difference, average reward per rollout: ", percent_error_average_reward, "\n")


        print(" Average RMSE action error per rollout: ", np.mean(action_errors))
        print("Average Action Taken by Expert: ", average_action_expert)
        print("% Action Error: ", percent_error_average_action, "\n")
        print("Total imitator steps: ", total_imitator_steps)
        print("Total expert steps: ", total_expert_steps)


    def save_controller_network(self):
        """
        Saves a keras tensorflow model to the specified path given in the command line params. Path must end with .h5.
        """
        print("Saving tensorflow model to: ", self.params['save_path'])
        self.action_network.save_network(self.params['save_path'])

    def save_controller_for_PPO(self):
        """
        Creates and saves a keras tensorflow model for training PPO with weights learned from imitation, to the specified path given in the command line params. Path must end with .h5.
        """
        self.action_network.save_network_PPO(self.params['save_path'])
