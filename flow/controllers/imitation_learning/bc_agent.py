"""Agent / policy to learn from expert demonstrations. 
    Underlying object of the policy is a MLP / NN"""


import numpy as np
import tensorflow as tf
import time


from flow.controllers.imitation_learning.replay_buffer import ReplayBuffer
from flow.controllers.imitation_learning.MLPPolicy import MLPPolicy

class BC_Agent:
    def __init__(self, sess, env, agent_params):

        # init vars
        self.env = env
        self.sess = sess
        self.agent_params = agent_params

        # actor/policy
        self.actor = MLPPolicy(sess,
                               self.agent_params['ac_dim'],
                               self.agent_params['ob_dim'],
                               self.agent_params['n_layers'],
                               self.agent_params['size'],
                               discrete = self.agent_params['discrete'],
                               learning_rate = self.agent_params['learning_rate'],
                               ) ## TODO: look in here and implement this

        # replay buffer
        self.replay_buffer = ReplayBuffer(self.agent_params['max_replay_buffer_size'])        
