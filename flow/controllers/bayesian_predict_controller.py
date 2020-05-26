from flow.controllers.base_controller import BaseController
from examples.sumo.bayesian_1_runner import bayesian_1_example

import numpy as np

class BayesianPredictController(BaseController):
    def __init__(self, veh_id, car_following_params):
        BaseController.__init__(self, veh_id, car_following_params, delay=1.0)

        self.query_env = bayesian_1_example(pedestrians=True).env
        self.query_env.reset()
        
        # Hacks to make fewer changes to bayesian_1_example
        self.query_env.k.vehicle.set_acc_controller('agent_0', (BayesianManualController, {}))
        self.query_env.k.vehicle.set_controlled('agent_0')

        # Convert between naming of vehicles across env and query_env
        self.map_vehicles = {
                    "human_0" : "human_0_0",
                    "human_1" : "human_0_1",
                    "agent_0" : "agent_0"
                }

    def get_accel(self, env):

        # Set query_env state to the same as env
        for veh_id in env.k.vehicle.get_ids():
            x, y = env.k.vehicle.get_xy(veh_id)
            edge_id = env.k.vehicle.get_edge(veh_id)
            lane = env.k.vehicle.get_lane(veh_id)

            self.query_env.k.vehicle.set_xy(
                    self.map_vehicles[veh_id],
                    edge_id, lane, x, y)

            self.query_env.k.vehicle.set_speed(
                    self.map_vehicles[veh_id],
                    env.k.vehicle.get_speed(veh_id))

        for ped_id in env.k.pedestrian.get_ids():
            x, y = env.k.pedestrian.get_xy(ped_id)
            edge = env.k.pedestrian.get_edge(ped_id)
            self.query_env.k.pedestrian.set_xy(
                    ped_id, edge, x, y)

        # Perform recursive look ahead
        action, _ = self.look_ahead()
        return action

    def look_ahead(self, steps=3):
        '''
        return (best accel, score of best accel)
        '''

        # Set score that vehicle tries to maximize
        score = self.query_env.k.vehicle.get_position('agent_0')
        if score > 25:
            score = -1

        # base case or colission(negative score)
        if steps == 0 or score < 0:
            return 0, score

        # Different accelerations to iterate over
        accels = np.arange(-2, 3) * 1.7 # TODO:add as a param

        # save current state information
        states = {}
        ped_states = {}
        for veh_id in self.query_env.k.vehicle.get_ids():
            x, y = self.query_env.k.vehicle.get_xy(veh_id)
            edge_id = self.query_env.k.vehicle.get_edge(veh_id)
            lane = self.query_env.k.vehicle.get_lane(veh_id)
            states[veh_id] = [x, y, edge_id, lane,
                    self.query_env.k.vehicle.get_speed(veh_id)]
        for ped_id in self.query_env.k.pedestrian.get_ids():
            x, y = self.query_env.k.pedestrian.get_xy(ped_id)
            edge_id = self.query_env.k.pedestrian.get_edge(ped_id)
            ped_states[ped_id] = [x, y, edge_id]

        best_action = 0
        best_score = 0

        # Iterate through each acceleration
        for a in accels:

            # Forward step
            self.query_env.k.vehicle.get_acc_controller('agent_0').set_accel(a)
            self.query_env.step(None)
            _, score = self.look_ahead(steps - 1)

            # Update if best accel so far
            if score >= best_score:
                best_score = score
                best_action = a

            # Restore query_env to before the forward step was taken
            for veh_id in self.query_env.k.vehicle.get_ids():
                state = states[veh_id]
                self.query_env.k.vehicle.set_xy(veh_id, state[2],
                        state[3], state[0], state[1])
                self.query_env.k.vehicle.set_speed(veh_id, state[4])

            for ped_id in self.query_env.k.pedestrian.get_ids():
                state = ped_states[ped_id]
                self.query_env.k.pedestrian.set_xy(ped_id, state[2],
                        state[0], state[1])

            # Take a step to reset vehicle positions and speed
            self.query_env.step(None)

        return best_action, best_score

    def get_action(self, env):
        # copied from parent and modified to maintain control through intersection
        """Convert the get_accel() acceleration into an action.

        If no acceleration is specified, the action returns a None as well,
        signifying that sumo should control the accelerations for the current
        time step.

        This method also augments the controller with the desired level of
        stochastic noise, and utlizes the "instantaneous" or "safe_velocity"
        failsafes if requested.

        Parameters
        ----------
        env : flow.envs.Env
            state of the environment at the current time step

        Returns
        -------
        float
            the modified form of the acceleration
        """
        # this is to avoid abrupt decelerations when a vehicle has just entered
        # a network and it's data is still not subscribed
        if len(env.k.vehicle.get_edge(self.veh_id)) == 0:
            return None

        accel = self.get_accel(env)

        # if no acceleration is specified, let sumo take over for the current
        # time step
        if accel is None:
            return None

        # add noise to the accelerations, if requested
        if self.accel_noise > 0:
            accel += np.random.normal(0, self.accel_noise)

        # run the failsafes, if requested
        if self.fail_safe == 'instantaneous':
            accel = self.get_safe_action_instantaneous(env, accel)
        elif self.fail_safe == 'safe_velocity':
            accel = self.get_safe_velocity_action(env, accel)

        return accel


class BayesianManualController(BaseController):
    def __init__(self, veh_id, car_following_params):
        BaseController.__init__(self, veh_id, car_following_params, delay=1.0)
        self.accel = None

    def set_accel(self, accel):
        self.accel = accel

    def get_accel(self, env):
        accel = self.accel
        self.accel = None
        return accel
