from flow.envs.multiagent import Bayesian0NoGridEnv

"""This env overwrites the state so that there is an imagined pedestrian that the l2 vehicle can see but l1 can't"""
class BayesianL2CooperativeEnv(Bayesian0NoGridEnv):

    """We have to modify this get_state function so that a few things are changed:
        1. The AVs are in control the whole
        """
    # def get_state(self):
    #     pass

    def compute_reward(self, rl_actions, **kwargs):
        reward = 0
        if self.k.vehicle.get_position('rl_0') > 40.0:
            reward = -10
        reward_dict = {}
        for veh_id in self.k.vehicle.get_ids():
            if reward == -10:
                reward_dict[veh_id] = reward
            else:
                speed = self.k.vehicle.get_speed(veh_id)
                if speed > 20.0:
                    reward_dict[veh_id] = -1
                else:
                    reward_dict[veh_id] = self.k.vehicle.get_speed(veh_id) / 10.0
        return reward_dict

class BayesianL2CooperativeEnvWithQueryEnv(BayesianL2CooperativeEnv):
    def __init__(self, env_params, sim_params, network, simulator='traci'):
        super().__init__(env_params, sim_params, network, simulator)
        self.query_env = None

    def reset(self):
        obs = super().reset()
        self.query_env.reset()
        return obs