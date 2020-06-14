import time
import numpy as np
import os


class ReplayBuffer(object):
    """ Replay buffer class to store state, action, expert_action, reward, next_state, terminal, state_info tuples.
    
    State info is a dict containing information on the vehicle whose observation that the s, a, r, t tuple is based on"""

    def __init__(self, max_size=100000):

        # max size of buffer
        self.max_size = max_size

        # store each rollout
        self.rollouts = []

        # store component arrays from each rollout
        self.observations = None
        self.actions = None
        self.expert_actions = None
        self.rewards = None
        self.next_observations = None
        self.terminals = None
        self.state_infos = None


    def add_rollouts(self, rollouts_list):
        """
        Add a list of rollouts to the replay buffer

        Parameters
        __________
        rollouts_list: list
            list of rollout dictionaries

        """

        for rollout in rollouts_list:
            self.rollouts.append(rollout)

        observations, actions, expert_actions, rewards, next_observations, terminals, state_infos = self.unpack_rollouts(rollouts_list)
        assert (not np.any(np.isnan(expert_actions))), "Invalid actions added to replay buffer"

        # only keep max_size tuples in buffer
        if self.observations is None:
            self.observations = observations[-self.max_size:]
            self.actions = actions[-self.max_size:]
            self.expert_actions = expert_actions[-self.max_size:]
            self.rewards = rewards[-self.max_size:]
            self.next_observations = next_observations[-self.max_size:]
            self.terminals = terminals[-self.max_size:]
            self.state_infos = state_infos[-self.max_size:]
        else:
            self.observations = np.concatenate([self.observations, observations])[-self.max_size:]
            self.actions = np.concatenate([self.actions, actions])[-self.max_size:]
            self.expert_actions = np.concatenate([self.expert_actions, expert_actions])[-self.max_size:]
            self.rewards = np.concatenate([self.rewards, rewards])[-self.max_size:]
            self.next_observations = np.concatenate([self.next_observations, next_observations])[-self.max_size:]
            self.terminals = np.concatenate([self.terminals, terminals])[-self.max_size:]
            self.state_infos = np.concatenate([self.state_infos, state_infos])[-self.max_size:]

    def sample_batch(self, batch_size):
        """
        Sample a batch of data (with size batch_size) from replay buffer.

        Parameters
        ----------
        batch_size: int
            size of batch to sample
        
        Returns
        _______
        Data in separate numpy arrays of observations, actions, and expert actions
        """
        assert self.observations is not None and self.actions is not None and self.expert_actions is not None

        size = len(self.observations)
        rand_inds = np.random.randint(0, size, batch_size)

        return self.observations[rand_inds], self.actions[rand_inds], self.expert_actions[rand_inds], self.state_infos[rand_inds]



    def unpack_rollouts(self, rollouts_list):
        """
        Convert list of rollout dictionaries to individual observation, action, rewards, next observation, terminal arrays
        Parameters
        ----------
        rollouts: list
            list of rollout dictionaries

        Returns
        ----------
        separate numpy arrays of observations, actions, rewards, next_observations, and is_terminals
        """
        observations = np.concatenate([rollout["observations"] for rollout in rollouts_list])
        actions = np.concatenate([rollout["actions"] for rollout in rollouts_list])
        expert_actions = np.concatenate([rollout["expert_actions"] for rollout in rollouts_list])
        rewards = np.concatenate([rollout["rewards"] for rollout in rollouts_list])
        next_observations = np.concatenate([rollout["next_observations"] for rollout in rollouts_list])
        terminals = np.concatenate([rollout["terminals"] for rollout in rollouts_list])
        state_infos = np.concatenate([rollout["state_infos"] for rollout in rollouts_list])


        return observations, actions, expert_actions, rewards, next_observations, terminals, state_infos
        
