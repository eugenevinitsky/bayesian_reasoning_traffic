"""Contains an experiment class for running bayesian simulations."""

import logging
import datetime
import numpy as np
import time
import os

from flow.core.experiment import Experiment
from flow.core.util import emission_to_csv
import csv


class Bayesian0Experiment(Experiment):

    def __init__(self, env):
        super().__init__(env)

    def run(self, num_runs, num_steps, rl_actions=None, convert_to_csv=False, collect_data=True):
        """
        See parent class
        """
        if rl_actions is None:
            def rl_actions(*_):
                return None
        
        state_data = []
        for i in range(num_runs):
            vel = np.zeros(num_steps)
            logging.info("Iter #" + str(i))
            ret = 0
            ret_list = []
            state = self.env.reset()
            for j in range(num_steps):
                # state is returned as an array
                state, reward, done, _ = self.env.step(rl_actions(state))
                # import ipdb; ipdb.set_trace()
                state_data.append(state[0]['human_0_0'])
        with open('test.csv', 'w', newline='') as fp:
            writer = csv.writer(fp, delimiter=',')
            # writer.writerow(["Self data", "is_ped", "foo"])  # write header
            for state in state_data:
                
                writer.writerows(state)
                print(state)
        return

