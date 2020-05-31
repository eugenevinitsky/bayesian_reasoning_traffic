"""Class to store agent's experiences e_t = (s_t, a_t, r_t, s_{t+1}), to be used in training"""

import time
import numpy as np
import tensorflow as tf
import gym
import os

from cs285.infrastructure.utils import *

class ReplayBuffer(object):

    def __init__(self, max_size=1000000):

        self.max_size = max_size

        # store each rollout
        self.paths = []

        # store (concatenated) component arrays from each rollout
        self.obs = None
        self.acs = None
        self.rews = None
        self.next_obs = None
        self.terminals = None

    def __len__(self):
        if self.obs:
            return self.obs.shape[0]
        else:
            return 0

    def add_rollouts(self, paths, concat_rew=True):

        # add new rollouts into our list of rollouts
        for path in paths:
            self.paths.append(path)

        # convert new rollouts into their component arrays, and append them onto our arrays
        observations, actions, rewards, next_observations, terminals = convert_listofrollouts(paths, concat_rew)

        if self.obs is None:
            self.obs = observations[-self.max_size:]
            self.acs = actions[-self.max_size:]
            self.rews = rewards[-self.max_size:]
            self.next_obs = next_observations[-self.max_size:]
            self.terminals = terminals[-self.max_size:]
        else:
            self.obs = np.concatenate([self.obs, observations])[-self.max_size:]
            self.acs = np.concatenate([self.acs, actions])[-self.max_size:]
            if concat_rew:
                self.rews = np.concatenate([self.rews, rewards])[-self.max_size:]
            else:
                if isinstance(rewards, list):
                    self.rews += rewards
                else:
                    self.rews.append(rewards)
                self.rews = self.rews[-self.max_size:]
            self.next_obs = np.concatenate([self.next_obs, next_observations])[-self.max_size:]
            self.terminals = np.concatenate([self.terminals, terminals])[-self.max_size:]

    ########################################
    ########################################

    def sample_random_data(self, batch_size):
        assert self.obs.shape[0] == self.acs.shape[0] == self.rews.shape[0] == self.next_obs.shape[0] == self.terminals.shape[0]
        assert self.obs.shape[0] >= batch_size
        sample_lst = np.array(range(batch_size))
        sample_lst = np.random.permutation(sample_lst)
        random_indices = sample_lst[:batch_size]

        return self.obs[random_indices], self.acs[random_indices], self.rews[random_indices], self.next_obs[random_indices], self.terminals[random_indices]

    def sample_recent_data(self, batch_size=1):
        return self.obs[-batch_size:], self.acs[-batch_size:], self.rews[-batch_size:], self.next_obs[-batch_size:], self.terminals[-batch_size:]


    def convert_listofrollouts(paths, concat_rew=True):
        """
        Take a list of rollout dictionaries
        and return separate arrays,
        where each array is a concatenation of that array from across the rollouts
        """
        observations = np.concatenate([path["observation"] for path in paths])
        actions = np.concatenate([path["action"] for path in paths])
        if concat_rew:
            rewards = np.concatenate([path["reward"] for path in paths])
        else:
            rewards = [path["reward"] for path in paths]
        next_observations = np.concatenate([path["next_observation"] for path in paths])
        terminals = np.concatenate([path["terminal"] for path in paths])
        return observations, actions, rewards, next_observations, terminals