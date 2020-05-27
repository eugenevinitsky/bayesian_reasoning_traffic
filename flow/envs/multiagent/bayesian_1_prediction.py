import numpy as np
from flow.envs.multiagent.bayesian_1_env import Bayesian1Env

class Bayesian1Prediction(Bayesian1Env):

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        super().__init__(env_params, sim_params, network, simulator)
        query_env = Bayesian1Env(env_params, sim_params, network, simulator)
