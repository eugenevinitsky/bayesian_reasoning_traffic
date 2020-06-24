from flow.controllers.base_controller import BaseController
# from examples.sumo.bayesian_1_runner import bayesian_1_example as query_env_generator

import numpy as np

class BayesianPredictController(BaseController):
    def __init__(self, veh_id, car_following_params, look_ahead_len=3):
        BaseController.__init__(self, veh_id, car_following_params, delay=1.0)

        self.look_ahead_len = look_ahead_len

    def get_accel(self, env):

        if not isinstance(env.query_env.k.vehicle.get_acc_controller('av_0'), BayesianManualController):
            env.query_env.k.vehicle.kernel_api.vehicle.setSpeedMode('av_0', 0)
            env.query_env.k.vehicle.set_acc_controller('av_0', (BayesianManualController, {}))
            env.query_env.k.vehicle.set_controlled('av_0')

        # Set query_env state to the same as env
        for veh_id in env.k.vehicle.get_ids():
            x, y = env.k.vehicle.get_xy(veh_id)
            edge_id = env.k.vehicle.get_edge(veh_id)
            lane = env.k.vehicle.get_lane(veh_id)

            env.query_env.k.vehicle.set_xy(
                    veh_id,
                    edge_id, lane, x, y)

            env.query_env.k.vehicle.set_speed(
                    veh_id,
                    env.k.vehicle.get_speed(veh_id))

        for ped_id in env.k.pedestrian.get_ids():
            x, y = env.k.pedestrian.get_xy(ped_id)
            edge = env.k.pedestrian.get_edge(ped_id)
            env.query_env.k.pedestrian.set_xy(
                    ped_id, edge, x, y)
        # Perform recursive look ahead
        action, _ = self.look_ahead(env, self.look_ahead_len)
        return action

    def store_info(self, env):
        # save current state information
        states = {}
        ped_states = {}
        for veh_id in env.k.vehicle.get_ids():
            x, y = env.k.vehicle.get_xy(veh_id)
            edge_id = env.k.vehicle.get_edge(veh_id)
            lane = env.k.vehicle.get_lane(veh_id)
            states[veh_id] = [x, y, edge_id, lane,
                    env.k.vehicle.get_speed(veh_id)]
        for ped_id in env.k.pedestrian.get_ids():
            x, y = env.k.pedestrian.get_xy(ped_id)
            edge_id = env.k.pedestrian.get_edge(ped_id)
            ped_states[ped_id] = [x, y, edge_id]
        return states, ped_states

    # TODO(@ev) return sum of rewards, not reward at leaf
    def look_ahead(self, env, steps):
        '''
        return (best accel, score of best accel)
        '''

        # Set score that vehicle tries to maximize
        score = self.compute_reward(env)

        near_exit = self.check_near_exit(env.query_env)
        # base case or collision (negative score)
        if steps == 0 or score < 0 or near_exit:
            return 0, score

        # Different accelerations to iterate over
        accels = [-4.5, 0, 2.6] # TODO:add as a param

        # save current state information
        states, ped_states = self.store_info(env.query_env)

        best_action = 0
        best_score = 0

        # Iterate through each acceleration
        for a in accels:

            # Forward step
            env.query_env.k.vehicle.get_acc_controller('av_0').set_accel(a)
            env.query_env.step(None)
            _, score = self.look_ahead(env, steps - 1)

            # Update if best accel so far
            if score >= best_score:
                best_score = score
                best_action = a

            # Restore query_env to before the forward step was taken
            for veh_id in env.query_env.k.vehicle.get_ids():
                state = states[veh_id]
                env.query_env.k.vehicle.set_xy(veh_id, state[2],
                        state[3], state[0], state[1])
                env.query_env.k.vehicle.set_speed(veh_id, state[4])

            for ped_id in env.query_env.k.pedestrian.get_ids():
                state = ped_states[ped_id]
                env.query_env.k.pedestrian.set_xy(ped_id, state[2],
                        state[0], state[1])

            # Take a step to reset vehicle positions and speed
            env.query_env.step(None)

        for ped_id in env.k.pedestrian.get_ids():
            x, y = env.k.pedestrian.get_xy(ped_id)
            edge_id = env.k.pedestrian.get_edge(ped_id)
            print([x, y, edge_id])
        for ped_id in env.query_env.k.pedestrian.get_ids():
            x, y = env.query_env.k.pedestrian.get_xy(ped_id)
            edge_id = env.query_env.k.pedestrian.get_edge(ped_id)
            print([x, y, edge_id])

        return best_action, best_score

    def compute_reward(self, env):
        """See class definition."""

        # TODO(@evinitsky) pick the right reward
        collision_vehicles = env.query_env.k.simulation.get_collision_vehicle_ids()
        collision_pedestrians = env.query_env.k.vehicle.get_pedestrian_crash(self.veh_id, env.query_env.k.pedestrian)

        if len(collision_pedestrians) > 0:
            reward = -1000
        elif self.veh_id in collision_vehicles:
            reward = -1000
        # move forwards bonus
        else:
            reward = env.query_env.k.vehicle.get_speed(self.veh_id)

        return reward

    def get_action(self, env, allow_junction_control=True):
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

        # this allows the acceleration behavior of vehicles in a junction be
        # described by sumo instead of an explicit model - for imitation learning,
        # we want out imitation controller to control the vehicle everywhere
        if env.k.vehicle.get_edge(self.veh_id)[0] == ":" and not allow_junction_control:
            return None

        # No control once you're past the intersection
        if self.on_final_edge(env) or self.before_intersection(env):
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

    def check_near_exit(self, env):
        """Check if the vehicle is near the end of the final edge. If so, don't let it go any further or bugs will ensue"""
        final_edge = env.k.vehicle.get_route(self.veh_id)[-1]
        edge = env.k.vehicle.get_edge(self.veh_id)
        edge_len = env.k.network.edge_length(final_edge)
        position = env.k.vehicle.get_position(self.veh_id)
        if edge == final_edge and  position > edge_len - 10:
            return True
        else:
            return False

    def on_final_edge(self, env):
        final_edge = env.k.vehicle.get_route(self.veh_id)[-1]
        edge = env.k.vehicle.get_edge(self.veh_id)
        if final_edge == edge and env.k.vehicle.get_position(self.veh_id) > 10.0:
            return True
        else:
            return False

    def before_intersection(self, env):
        """Check that we are more than five meters from the intersection"""
        first_edge = env.k.vehicle.get_route(self.veh_id)[0]
        edge = env.k.vehicle.get_edge(self.veh_id)
        if first_edge == edge and env.k.vehicle.get_position(self.veh_id) < 45.0:
            return True
        else:
            return False

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
