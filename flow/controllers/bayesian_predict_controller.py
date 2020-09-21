from copy import copy
from flow.controllers.base_controller import BaseController
from itertools import combinations_with_replacement
# from examples.sumo.bayesian_1_runner import bayesian_1_example as query_env_generator

import numpy as np

class BayesianPredictController(BaseController):
    def __init__(self, veh_id, car_following_params, look_ahead_len=4):
        BaseController.__init__(self, veh_id, car_following_params, delay=1.0)

        self.look_ahead_len = look_ahead_len
        self.accel = None

    def sync_envs(self, env):
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

            # TODO(KL) may need to reset the priors of the QUERY env

        for ped_id in env.k.pedestrian.get_ids():
            x, y = env.k.pedestrian.get_xy(ped_id)
            edge = env.k.pedestrian.get_edge(ped_id)
            env.query_env.k.pedestrian.set_xy(
                    ped_id, edge, x, y)

        if hasattr(env, 'arrived_ids'):
            env.query_env.arrived_ids = env.arrived_ids

        if hasattr(env, 'inside_intersection'):
            env.query_env.inside_intersection = env.inside_intersection

        if hasattr(env, 'got_to_intersection'):
            env.query_env.got_to_intersection = env.got_to_intersection

        for veh_id in env.k.vehicle.get_ids():
            controller = env.k.vehicle.get_acc_controller(veh_id)
            if hasattr(controller, 'priors'):
                query_controller = env.query_env.k.vehicle.get_acc_controller(veh_id)
                # reset it because we would never know what the priors are
                query_controller.priors = {} #copy(controller.priors)

        # env.query_env.k.vehicle.get_acc_controller()

        env.query_env.step(None)

    def get_accel(self, env, ped_prob=1):
        # import ipdb; ipdb.set_trace()
        # TODO(KL) what's the point of this?
        if not isinstance(env.query_env.k.vehicle.get_acc_controller(self.veh_id), BayesianManualController):
            env.query_env.k.vehicle.kernel_api.vehicle.setSpeedMode(self.veh_id, 0)
            env.query_env.k.vehicle.set_acc_controller(self.veh_id, (BayesianManualController, {}))
            # env.query_env.k.vehicle.set_controlled(self.veh_id) # TODO KL try duplicating this?

        # Set query_env state to the same as env
        self.sync_envs(env)

        # # Remove pedestrians from query_env
        # ped_states = {}
        # for ped_id in env.query_env.k.pedestrian.get_ids():
        #     ped_state = {}
        #     ped_state['edge_id'] = env.k.pedestrian.get_edge(ped_id)
        #     ped_state['position'] = env.k.pedestrian.get_lane_position(ped_id)
        #     ped_state['stage'] = env.k.pedestrian.kernel_api.person.getStage(ped_id)
        #     ped_states[ped_id] = ped_state
        #     env.query_env.k.pedestrian.remove(ped_id)

        # # Look ahead with no pedestrians
        # _, _, action_scores_no_ped = self.look_ahead(env, self.look_ahead_len)

        # # Add pedestrians back in to query_env and sync
        # for ped_id in ped_states:
        #     env.query_env.k.kernel_api.person.add(
        #             ped_id,
        #             ped_states[ped_id]['edge_id'],
        #             ped_states[ped_id]['position'])
        #     # TODO(@evinitsky) figure out what this is all about 
        # TODO @ euguene this is for p(look ahead score with ped) + (1-p)(look ahead score without ped)
        #     env.query_env.k.kernel_api.person.appendStage(ped_id,
        #             ped_states[ped_id]['stage'])

        # env.query_env.step(None)
        # self.sync_envs(env)

        # Perform recursive look ahead
        best_action_sequence, best_score, action_scores_ped = self.look_ahead(env, self.look_ahead_len)

        # Compute weighted average for scores to determine best action
        # best_action = (0.0) * self.look_ahead_len
        # best_score = -1e6
        # for a in action_scores_ped:
        #     score = (ped_prob * action_scores_ped[a]) + \
        #             ((1 - ped_prob) * action_scores_no_ped[a])
        #     if score >= best_score:
        #         best_score = score
        #         best_action = a

        # store it for replay
        self.accel = best_action_sequence[0]
        return best_action_sequence[0]

    def store_info(self, env):
        # save current state information
        states = {}
        ped_states = {}
        controllers = {}
        for veh_id in env.k.vehicle.get_ids():
            x, y = env.k.vehicle.get_xy(veh_id)
            edge_id = env.k.vehicle.get_edge(veh_id)
            lane = env.k.vehicle.get_lane(veh_id)
            states[veh_id] = [x, y, edge_id, lane,
                    env.k.vehicle.get_speed(veh_id)]
            controller = env.k.vehicle.get_acc_controller(veh_id)
            controllers[veh_id] = controller
        for ped_id in env.k.pedestrian.get_ids():
            x, y = env.k.pedestrian.get_xy(ped_id)
            edge_id = env.k.pedestrian.get_edge(ped_id)
            ped_states[ped_id] = [x, y, edge_id]
        return states, ped_states, controllers

    # TODO(@ev) return sum of rewards, not reward at leaf
    def look_ahead(self, env, steps):
        '''
        return (best accel, score of best accel, dict accel:score)
        '''

        # Set score that vehicle tries to maximize
        # score = self.compute_reward(env)

        near_exit = self.check_near_exit(env.query_env)
        # base case or collision (negative score)
        # if steps == 0 or score < 0 or near_exit:

        # don't do any control once you're near the exit
        if near_exit:
            return 0, 0, {}

        # Different accelerations to iterate over
        accels = [-4.5, 2.6] # TODO:add as a param

        # save current state information
        states, ped_states, controllers = self.store_info(env)

        action_scores = {}
        best_action = [0, 0, 0]
        best_score = -1e6

        # Iterate through each acceleration
        for action_comb in combinations_with_replacement(accels, steps):
            # print(list(combinations_with_replacement(accels, steps)))
            # import ipdb; ipdb.set_trace()
            score_total = 0
            for a in action_comb:

                # Forward step
                # figure out who'se getting controlled ??
                # import ipdb; ipdb.set_trace()
                env.query_env.k.vehicle.get_acc_controller(self.veh_id).set_accel(a)
                env.query_env.step(None)
                score = self.compute_reward(env)
                score_total += score

                # Update if best accel so far
                # action_scores[a] = score
            # if steps == 3:
            #     print('score of action {} is {}'.format(a, score))
            #     print('speed of vehicle is {}'.format(env.query_env.k.vehicle.get_speed('temp_0')))
            # print(f'action_comb is {action_comb}, score_total is {score_total}')
            # break ties by going slower earlier
            if (score_total == best_score and action_comb[0] < best_action[0]) or score_total > best_score:
                best_score = score_total
                best_action = action_comb

            # Restore query_env to before the forward step was taken
            for veh_id in env.query_env.k.vehicle.get_ids():
                state = states[veh_id]
                env.query_env.k.vehicle.set_xy(veh_id, state[2],
                        state[3], state[0], state[1])
                env.query_env.k.vehicle.set_speed(veh_id, state[4])
                if veh_id != self.veh_id:
                    env.query_env.k.vehicle.set_controller_directly(veh_id, controllers[veh_id])

            for ped_id in env.query_env.k.pedestrian.get_ids():
                state = ped_states[ped_id]
                env.query_env.k.pedestrian.set_xy(ped_id, state[2],
                        state[0], state[1])

            # Take a step to reset vehicle positions and speed
            env.query_env.step(None)

        # make sure the query env has the right action
        env.query_env.k.vehicle.get_acc_controller(self.veh_id).accel = best_action[0] # might need set accel? TODO(KL)

        return best_action, best_score, action_scores

    def compute_reward(self, env):
        """See class definition."""

        # TODO(@evinitsky) pick the right reward
        # collision_vehicles = env.query_env.k.simulation.get_collision_vehicle_ids()
        # collision_pedestrians = False
        # for veh_id in env.k.vehicle.get_ids():
        #     if env.query_env.k.vehicle.get_pedestrian_crash(veh_id, env.query_env.k.pedestrian):
        #         collision_pedestrians = True
        #
        # if collision_pedestrians:
        #     reward = -1000
        # elif len(collision_vehicles) > 0:
        #     reward = -1000
        # # move forwards bonus
        # else:
        #     reward = env.query_env.k.vehicle.get_speed(self.veh_id)

        reward = env.query_env.compute_reward(rl_actions=None)[self.veh_id]
        # break ties by picking going faster
        if reward == 0:
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
        # if self.on_final_edge(env) or self.before_intersection(env):
        #     return None
        if self.on_final_edge(env):
            return None

        accel = self.get_accel(env)
        if str(env) == "<BayesianL2CooperativeEnvWithQueryEnv instance>":
            # import ipdb; ipdb.seqt_trace()
            print('selected action of temp_0 (red) is ', accel)

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
        BaseController.__init__(self, veh_id, car_following_params, delay=0.0)
        self.accel = None

    def set_accel(self, accel):
        self.accel = accel

    def get_accel(self, env):
        accel = self.accel
        self.accel = None
        return accel

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
        # if self.on_final_edge(env) or self.before_intersection(env):
        #     return None
        if self.on_final_edge(env):
            return None

        # import ipdb; ipdb.set_trace()
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

    def on_final_edge(self, env):
        final_edge = env.k.vehicle.get_route(self.veh_id)[-1]
        edge = env.k.vehicle.get_edge(self.veh_id)
        if final_edge == edge and env.k.vehicle.get_position(self.veh_id) > 10.0:
            return True
        else:
            return False