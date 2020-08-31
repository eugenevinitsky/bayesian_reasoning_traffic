"""Environment testing scenario one of the bayesian envs."""
from collections import defaultdict
from copy import deepcopy, copy
import math
import numpy as np
from gym.spaces import Box, Discrete
from flow.core.params import SumoCarFollowingParams
from flow.controllers import RuleBasedIntersectionController
from flow.envs.multiagent.base import MultiEnv

from traci.exceptions import FatalTraCIError
from traci.exceptions import TraCIException
from flow.utils.exceptions import FatalFlowError
from bayesian_inference.get_inferer import get_inferrer
from bayesian_inference.inference import get_filtered_posteriors


# TODO(KL) means KL's reminder for KL

ADDITIONAL_ENV_PARAMS = {
    # maximum acceleration of autonomous vehicles
    'max_accel': 2.6,
    # maximum deceleration of autonomous vehicles
    'max_decel': 4.5,
    # desired velocity for all vehicles in the network, in m/s
    "target_velocity": 25,
    # how many objects in our local radius we want to return
    "max_num_objects": 3,
    # how large of a radius to search in for a given vehicle in meters
    "search_veh_radius": 50,
    # how large of a radius to search for pedestrians in for a given vehicle in meters (create effect of only seeing pedestrian only when relevant)
    "search_ped_radius": 22,
    # whether or not we have a discrete action space,
    "discrete": True,
    # whether to randomize which edge the vehicles are coming from
    "randomize_vehicles": True,
    # whether to append the prior into the state
    "inference_in_state": False,
    # whether to grid the cone "search_veh_radius" in front of us into 6 grid cells
    "use_grid": False
}

HARD_BRAKE_PENALTY = 0.001
NUM_PED_LOCATIONS = 4
JUNCTION_ID = '(1.1)'
DISCRETE_VALS = 10


class Bayesian0NoGridEnv(MultiEnv):
    """Testing whether an agent can learn to navigate successfully crossing the env described
    in scenario 1 of Jakob's diagrams. Please refer to the sketch for more details. Basically,
    inferring that the human is going to cross allows one of the vehicles to succesfully cross.

    Required from env_params:

    * max_accel: maximum acceleration for autonomous vehicles, in m/s^2
    * max_decel: maximum deceleration for autonomous vehicles, in m/s^2
    * target_velocity: desired velocity for all vehicles in the network, in m/s

    The following states, actions and rewards are considered for one autonomous
    vehicle only, as they will be computed in the same way for each of them.

    Attributes
    ----------
    updated_probs_fn: function
        black-box policy the updated probabilities of pedestrians being in grid i
        inputs: action, the acceleration of vehicle
        six previous probabilities of pedestrian being in grid i
        
        (veh_id, action, non_ped_states, prev_probs) 

    States
        TBD

    Actions
        The action consists of an acceleration, bound according to the
        environment parameters.

    Rewards
        TBD

    Termination
        A rollout is terminated if the time horizon is reached or if two
        vehicles collide into one another.
    """

    # TODO(KL) Not sure how to feed in params to the _init_: the envs object is created in registry.py (??)  Hard
    def __init__(self, env_params, sim_params, network, simulator='traci', ):
        # for p in ADDITIONAL_ENV_PARAMS.keys():
        #     if p not in env_params.additional_params:
        #         raise KeyError(
        #             'Environment parameter "{}" not supplied'.format(p))

        super().__init__(env_params, sim_params, network, simulator)
        self.discrete = env_params.additional_params.get("discrete", False)
        self.max_num_objects = env_params.additional_params.get("max_num_objects", 3)
        self.veh_obs_names = ["rel_x", "rel_y", "speed", "yaw", "arrive_before"]
        # setup information for the gridding if it is needed
        self.use_grid = env_params.additional_params.get("use_grid", False)
        self.priors = defaultdict(dict)
        if self.use_grid:
            self.num_grid_cells = 6
            self.self_obs_names = ["yaw", "speed", "turn_num", "curr_edge", "end_edge", "edge_pos", "veh_x", "veh_y"]
            self.ped_names = ["ped_in_0", "ped_in_1", "ped_in_2", "ped_in_3", "ped_in_4", "ped_in_5"]
        else:
            # last_seen this tracks when we last saw a vehicle so we don't forget it immediately
            self.self_obs_names = ["yaw", "speed", "turn_num", "curr_edge", "end_edge", "edge_pos", "veh_x", "veh_y",
                                   "arrival_pos",
                                   "last_seen"]
            self.ped_names = ["ped_in_0", "ped_in_1", "ped_in_2", "ped_in_3"]
        # list of tracked priors
        self.prior_names = ["infer_{}_{}".format(loc_id, veh_idx) for veh_idx in range(self.max_num_objects)
                            for loc_id in range(len(self.ped_names))]
        self.prior_probs = -1 * np.ones(len(self.prior_names))

        self.search_veh_radius = self.env_params.additional_params["search_veh_radius"]
        # self.search_ped_radius = self.env_params.additional_params["search_ped_radius"]
        # variable to encourage vehicle to move in curriculum training
        self.speed_reward_coefficient = 1
        # track all rl_vehicles: hack to compute the last reward of an rl vehicle (reward for arriving, set states to 0)
        self.rl_set = set()
        # track vehicles that have already been rewarded once for passing intersection
        self.past_intersection_rewarded_set = set()
        # feature for arrival
        self.arrival_order = {}
        # set to store vehicles currently inside the intersection
        self.inside_intersection = set()

        # dict to store the ground truth state of pedestrians at the four locations
        self.prev_loc_ped_state = {loc: 0 for loc in range(NUM_PED_LOCATIONS)}
        # dict to store the counts for each possible transiion
        self.ped_transition_cnt = {loc: {'00': 1, '01': 1, '10': 1, '11': 1} for loc in range(NUM_PED_LOCATIONS)}

        self.near_intersection_rewarded_set_1 = set()
        self.near_intersection_rewarded_set_2 = set()
        self.near_intersection_rewarded_set_3 = set()

        self.edge_to_num = {
            "(1.2)--(1.1)": 0,
            "(1.1)--(1.2)": 1,
            "(2.1)--(1.1)": 2,
            "(1.1)--(2.1)": 3,
            "(1.0)--(1.1)": 4,
            "(1.1)--(1.0)": 5,
            "(0.1)--(1.1)": 6,
            "(1.1)--(0.1)": 7
        }

        self.edge_to_int = {
            "(1.1)--(2.1)": 0,
            "(2.1)--(1.1)": 1,
            "(1.1)--(1.2)": 2,
            "(1.2)--(1.1)": 3,
            "(1.1)--(0.1)": 4,
            "(0.1)--(1.1)": 5,
            "(1.1)--(1.0)": 6,
            "(1.0)--(1.1)": 7
        }

        self.num_to_edge = {
            num: edge for edge, num in self.edge_to_num.items()
        }

        self.in_edges = ["(2.1)--(1.1)",
                         "(1.2)--(1.1)",
                         "(0.1)--(1.1)",
                         "(1.0)--(1.1)"]

        max_accel, max_decel = self.env_params.additional_params['max_accel'], -np.abs(
            self.env_params.additional_params['max_decel'])
        step_size = (max_accel - max_decel) / (DISCRETE_VALS - 1)
        self.discrete_actions_to_accels = [max_decel + i * step_size for i in range(DISCRETE_VALS)]
        # the space should always include zero for coasting
        self.discrete_actions_to_accels.append(0)

        # this is used for inference
        # wonder if it's better to specify the file path or the kind of policy (the latter?)
        self.inference_in_state = env_params.additional_params.get("inference_in_state", False)
        # TODO(@evinitsky) the inference code is not merged yet
        # if self.inference_in_state:
        #     path_to_inferrer = "/home/thankyou-always/TODO/research/bayesian_reasoning_traffic/flow/controllers/imitation_learning/model_files/c.h5"
        #     self.agent = get_inferrer(path=path_to_inferrer, inferrer_type="imitation")

        self.controller_dict = {}

    @property
    def observation_space(self):
        """See class definition."""
        max_objects = self.env_params.additional_params["max_num_objects"]
        # observations of your own state and other cars
        num_obs = len(self.self_obs_names) + len(self.ped_names) + max_objects * len(self.veh_obs_names)
        # belief variable over whether a pedestrian is there
        if self.inference_in_state:
            num_obs += len(self.prior_names)
        obs_space = Box(-float('inf'), float('inf'), shape=(num_obs,), dtype=np.float32)
        return obs_space

    @property
    def action_space(self):
        """See class definition."""
        if self.discrete:
            # 10 different accelerations plus 0
            return Discrete(DISCRETE_VALS + 1)
        else:
            return Box(
                low=-1,
                high=1,
                shape=(1,),  # (4,),
                dtype=np.float32)

    def _apply_rl_actions(self, rl_actions):
        """See class definition."""
        # in the warmup steps, rl_actions is None
        for rl_id in self.k.vehicle.get_rl_ids():
            if not self.arrived_intersection(rl_id) or self.past_intersection(rl_id):
                self.k.vehicle.set_speed_mode(rl_id, 'right_of_way')

        if rl_actions:
            rl_ids = []
            accels = []
            for rl_id, actions in rl_actions.items():
                # if rl_id in self.k.vehicle.get_rl_ids():

                if not self.arrived_intersection(rl_id):
                    continue

                self.k.vehicle.set_speed_mode(rl_id, 'aggressive')

                if self.discrete:
                    accel = self.discrete_actions_to_accels[actions]
                else:
                    max_decel = -np.abs(self.env_params.additional_params["max_decel"])
                    max_accel = self.env_params.additional_params["max_accel"]
                    accel = max_decel + (max_accel - max_decel) * (actions[0] + 1) / 2

                # if we are past the intersection, go full speed ahead but don't crash
                # if self.past_intersection(rl_id):
                #     self.k.vehicle.set_speed_mode(rl_id, 'right_of_way')
                #     continue

                rl_ids.append(rl_id)
                accels.append(accel)

            self.k.vehicle.apply_acceleration(rl_ids, accels)

    def arrived_intersection(self, veh_id):
        """Return True if vehicle is at or past the intersection and false if not."""
        intersection_length = self.net_params.additional_params['grid_array']['inner_length']
        dist_to_intersection = intersection_length - self.k.vehicle.get_position(veh_id)
        return not (len(self.k.vehicle.get_route(veh_id)) == 0 or \
                    self.k.vehicle.get_edge(veh_id) == self.k.vehicle.get_route(veh_id)[0] and \
                    dist_to_intersection > 2)

    def past_intersection(self, veh_id):
        """Return True if vehicle is at least 20m past the intersection (we had control back to SUMO at this point) & false if not"""  # TODO(KL)
        try:
            on_post_intersection_edge = len(self.k.vehicle.get_route(veh_id)) == 0 or self.k.vehicle.get_edge(veh_id) == \
                                        self.k.vehicle.get_route(veh_id)[-1]
        except:
            import ipdb;
            ipdb.set_trace()
            on_post_intersection_edge = self.k.vehicle.get_route(veh_id) == "" or self.k.vehicle.get_edge(veh_id) == \
                                        self.k.vehicle.get_route(veh_id)[-1]

        if on_post_intersection_edge and self.k.vehicle.get_position(
                veh_id) > 4:  # vehicle arrived at final destination, 8 is a random distance
            return True
        elif self.k.vehicle.get_edge(veh_id) == '':
            return True
        return False

    def state_for_id(self, rl_id):

        num_self_obs = len(self.self_obs_names)
        num_ped_obs = len(self.ped_names)
        num_veh_obs = len(self.veh_obs_names)
        self.update_intersection_state(rl_id)
        if 'av_0' == rl_id:
            # this is just used for tracking the reward of av_0 for tensorboards
            self.reward = {}
        self.rl_set.add(rl_id)
        # assert rl_id in self.arrival_order

        observation = -np.ones(self.observation_space.shape[0])

        visible_vehicles, visible_pedestrians, visible_lanes = self.find_visible_objects(rl_id,
                                                                                         self.search_veh_radius)

        # sort visible vehicles by angle where 0 degrees starts facing the right side of the vehicle
        # visible_vehicles = sorted(visible_vehicles, key=lambda v: \
        #     (self.k.vehicle.get_relative_angle(rl_id, \
        #                                        self.k.vehicle.get_orientation(v)[:2]) + 90) % 360)

        observation[:num_self_obs + num_ped_obs - 1] = self.get_self_obs(rl_id, visible_pedestrians, visible_lanes)
        # shift the peds over by 1 so that self obs all come first
        observation[num_self_obs + 1: num_self_obs + num_ped_obs + 1] = observation[num_self_obs: num_self_obs + num_ped_obs]
        if len(visible_vehicles) == 0:
            self.last_seen += 1 / 50.0
        else:
            self.last_seen = 0
        observation[num_self_obs - 1] = self.last_seen
        self.ped_variables = observation[num_self_obs: num_self_obs + num_ped_obs]

        veh_x, veh_y = self.k.vehicle.get_orientation(rl_id)[:2]

        # setting the 'arrival' order feature: 1 is if agent arrives before; 0 if agent arrives after

        # print('number of visible vehicles is ', len(visible_vehicles))
        # print('visible vehiclesares ', visible_vehicles)

        for index, veh_id in enumerate(visible_vehicles):
            before = self.arrival_position(veh_id)

            observed_yaw = self.k.vehicle.get_yaw(veh_id)
            observed_speed = self.k.vehicle.get_speed(veh_id)
            observed_x, observed_y = self.k.vehicle.get_orientation(veh_id)[:2]
            rel_x = observed_x - veh_x
            rel_y = observed_y - veh_y
            # print('rel_x: {}, rel_y: {} '.format(rel_x, rel_y))

            # Consider the first 3 visible vehicles, but we don't need to do inference for the humans, we only need a dummy state
            if index < self.max_num_objects and 'human' not in rl_id:
                observation[(index * num_veh_obs) + num_self_obs + num_ped_obs:
                            num_veh_obs * (index + 1) + num_self_obs + num_ped_obs] = \
                    [observed_yaw / 360, observed_speed / 20,
                     rel_x / 50, rel_y / 50, before / 5]
                if self.inference_in_state:
                    # only perform inference if the visible veh has arrived
                    if self.arrived_intersection(veh_id) and not self.past_intersection(veh_id):
                        dummy_obs = np.zeros(self.observation_space.shape[0])

                        # compute the action it did take. Note that we could store this but this is easier
                        _, visible_pedestrians, visible_lanes = self.find_visible_objects(veh_id, self.search_veh_radius)
                        ped_params = self.four_way_ped_params(visible_pedestrians, visible_lanes)
                        # TODO fix magic numbers
                        dummy_obs[[10, 11, 12, 13]] = ped_params
                        if hasattr(self.k.vehicle.get_acc_controller(veh_id), 'get_action_with_ped'):
                            acceleration = self.k.vehicle.get_acc_controller(veh_id).get_action(self)
                            controller = self.k.vehicle.get_acc_controller(veh_id)
                        else:
                            if veh_id not in self.controller_dict:
                                self.controller_dict[veh_id] = RuleBasedIntersectionController(veh_id,
                                                            car_following_params=SumoCarFollowingParams(
                                                                min_gap=2.5,
                                                                decel=7.5,
                                                                # avoid collisions at emergency stops
                                                                speed_mode="right_of_way",
                                                            )
                                                            )
                            acceleration = self.controller_dict[veh_id].get_action_with_ped(self,
                                                                                           dummy_obs)
                            controller = self.controller_dict[veh_id]

                        # now we can do the filtering
                        # we pass a zero of states because it's just a dummy obs, only the ped part of it affects the behavior
                        updated_ped_probs, self.priors[veh_id] = get_filtered_posteriors(self, controller,
                                                                                         acceleration,
                                                                                         np.zeros(self.observation_space.shape[0]),
                                                                                         self.priors.get(veh_id,
                                                                                                         {}),
                                                                                         veh_id)
                        veh_idx = int(veh_id.split('_')[-1])
                        # TODO(we are currently using the vehicle id as an index but that doesn't give us any way to correlate it
                        # to the other state variables; the index doesn't tell us which car is which
                        self.prior_probs[veh_idx * len(self.ped_names): (veh_idx + 1) * len(self.ped_names)] = updated_ped_probs
        if self.inference_in_state:
            prior_index = (self.max_num_objects * num_veh_obs) + num_self_obs + num_ped_obs
            observation[prior_index:] = self.prior_probs
            print(observation[prior_index:])
        return observation


    def get_state(self):
        """For a radius around the car, return the 3 closest objects with their X, Y position relative to you,
        their speed, a flag indicating if they are a pedestrian or not, and their yaw."""
        new_loc_states = self.curr_ped_state()
        for loc, val in enumerate(new_loc_states):
            prev = self.prev_loc_ped_state[loc]
            curr = val
            self.ped_transition_cnt[loc][f'{prev}{curr}'] += 1
            self.prev_loc_ped_state[loc] = curr
        obs = {}
        for veh_id in self.k.vehicle.get_ids():
            if veh_id not in self.arrival_order and self.arrived_intersection(veh_id):
                self.arrived_intersection(veh_id)
                self.arrival_order[veh_id] = len(self.arrival_order)

        veh_ids = self.k.vehicle.get_ids()
        rl_ids = self.k.vehicle.get_rl_ids()
        # print('rl ids are ', rl_ids)
        # avs are trained via DQN, rl is the L2 car. We have all these conditions so we can use pre-trained controllers later.
        valid_ids = [veh_id for veh_id in veh_ids if ('av' in veh_id or 'rl' in veh_id or veh_id in rl_ids)]
        # print('valid ids are ', valid_ids)
        for rl_id in valid_ids:
            if self.arrived_intersection(rl_id) and not self.past_intersection(
                    rl_id):
                obs.update({rl_id: self.state_for_id(rl_id)})
                # print('inside intersection', self.inside_intersection)
        return obs

    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        # in the warmup steps
        # if rl_actions is None:
        #     return {}

        rewards = {}
        veh_ids = self.k.vehicle.get_ids()

        # avs are trained via DQN, rl is the L2 car
        rl_ids = self.k.vehicle.get_rl_ids()
        # avs are trained via DQN, rl is the L2 car
        valid_ids = [veh_id for veh_id in veh_ids if ('av' in veh_id or 'rl' in veh_id or veh_id in rl_ids)]
        for rl_id in valid_ids:
            # reward rl slightly earlier than when control is given back to SUMO
            # if self.past_intersection(rl_id):
            #
            #     # good job on getting to goal and going fast. We keep these rewards tiny to not overwhelm the
            #     # pedestrian penalty
            #     rewards[rl_id] = 0.4 / 500.0
            #     continue

            if self.arrived_intersection(rl_id) and not self.past_intersection(rl_id):
                reward = 0
                self.reward[rl_id] = 0
                edge_pos = self.k.vehicle.get_position(rl_id)
                #
                # if 47 < edge_pos < 50 and self.k.vehicle.get_speed(rl_id) < 1.0:
                #     # this reward needs to be a good deal less than the "get to goal reward". You can't just sit here
                #     # and maximize your reward
                #     reward = 0.4 / 2000.0

                # TODO(@evinitsky) pick the right reward
                collision_vehicles = self.k.simulation.get_collision_vehicle_ids()
                collision_pedestrians = self.k.vehicle.get_pedestrian_crash(rl_id, self.k.pedestrian)

                if len(collision_pedestrians) > 0:
                    reward = self.env_params.additional_params.get("ped_collision_penalty", -10)
                elif rl_id in collision_vehicles:
                    reward = -10.0

                # # make the reward positive so you have no incentive to die
                # # reward += 0.4
                # reward /= 500

                # penalty for jerkiness
                # if rl_actions and rl_id in rl_actions.keys():
                #     if self.discrete:
                #         accel = self.discrete_actions_to_accels[rl_actions[rl_id]]
                #     else:
                #         accel = rl_actions[rl_id][0]
                #     reward += min(accel, 0) / 50.0
                # if inside_intersection and self.k.vehicle.get_speed(rl_id) < 1.0:
                #     reward -= 0.1

                # if reward < 0:
                #     import ipdb; ipdb.set_trace()
                rewards[rl_id] = reward
                self.reward[rl_id] = reward

        return rewards

    def step(self, rl_actions):
        """Advance the environment by one step.

        Assigns actions to autonomous and human-driven agents (i.e. vehicles,
        traffic lights, etc...). Actions that are not assigned are left to the
        control of the simulator. The actions are then used to advance the
        simulator by the number of time steps requested per environment step.

        Results from the simulations are processed through various classes,
        such as the Vehicle and TrafficLight kernels, to produce standardized
        methods for identifying specific network state features. Finally,
        results from the simulator are used to generate appropriate
        observations.

        Parameters
        ----------
        rl_actions : array_like
            an list of actions provided by the rl algorithm

        Returns
        -------
        observation : dict of array_like
            agent's observation of the current environment
        reward : dict of floats
            amount of reward associated with the previous state/action pair
        done : dict of bool
            indicates whether the episode has ended
        info : dict
            contains other diagnostic information from the previous action
        """
        # used to track rewards
        for _ in range(self.env_params.sims_per_step):
            self.time_counter += 1
            self.step_counter += 1

            # perform acceleration actions for controlled human-driven vehicles
            if len(self.k.vehicle.get_controlled_ids()) > 0:
                accel = []
                for veh_id in self.k.vehicle.get_controlled_ids():
                    accel_contr = self.k.vehicle.get_acc_controller(veh_id)
                    action = accel_contr.get_action(self)
                    accel.append(action)
                self.k.vehicle.apply_acceleration(
                    self.k.vehicle.get_controlled_ids(), accel)

            # perform lane change actions for controlled human-driven vehicles
            if len(self.k.vehicle.get_controlled_lc_ids()) > 0:
                direction = []
                for veh_id in self.k.vehicle.get_controlled_lc_ids():
                    target_lane = self.k.vehicle.get_lane_changing_controller(
                        veh_id).get_action(self)
                    direction.append(target_lane)
                self.k.vehicle.apply_lane_change(
                    self.k.vehicle.get_controlled_lc_ids(),
                    direction=direction)

            # perform (optionally) routing actions for all vehicle in the
            # network, including rl and sumo-controlled vehicles
            routing_ids = []
            routing_actions = []
            for veh_id in self.k.vehicle.get_ids():
                if self.k.vehicle.get_routing_controller(veh_id) is not None:
                    routing_ids.append(veh_id)
                    route_contr = self.k.vehicle.get_routing_controller(veh_id)
                    routing_actions.append(route_contr.choose_route(self))
            self.k.vehicle.choose_routes(routing_ids, routing_actions)

            self.apply_rl_actions(rl_actions)

            self.additional_command()

            # advance the simulation in the simulator by one step
            self.k.simulation.simulation_step()

            # store new observations in the vehicles and traffic lights class
            self.k.update(reset=False)

            # track observed IDs
            self.observed_rl_ids.update(self.k.vehicle.get_rl_ids())

            # update the colors of vehicles
            if self.sim_params.render:
                self.k.vehicle.update_vehicle_colors()

            # crash encodes whether the simulator experienced a collision
            # we only want to call crash if an RL car collides though
            crash = self.k.simulation.check_collision()
            collide_ids = []
            veh_crash = False
            if crash:
                collide_ids += self.k.simulation.get_collision_vehicle_ids()
                for veh_id in self.k.vehicle.get_rl_ids():
                    if veh_id in collide_ids:
                        veh_crash = True

            # update crash if there's an pedestrian-vehicle collision
            ped_crash = False
            if self.k.pedestrian:
                for veh_id in self.k.vehicle.get_rl_ids():
                    if len(self.k.vehicle.get_pedestrian_crash(veh_id, self.k.pedestrian)) > 0:
                        collide_ids.append(veh_id)
                        ped_crash = True
            # if crash:
            #     import ipdb; ipdb.set_trace()

            # stop collecting new simulation steps if there is a collision
            if veh_crash or ped_crash:
                break

        states = self.get_state()
        infos = {key: {} for key in states.keys()}

        # compute the reward
        if self.env_params.clip_actions:
            clipped_actions = self.clip_actions(rl_actions)
            reward = self.compute_reward(clipped_actions, fail=crash)
        else:
            reward = self.compute_reward(rl_actions, fail=crash)
        # if states.keys() != reward.keys():
        #     reward = self.compute_reward(rl_actions, fail=crash)
        #     states = self.get_state()
        done = {key: key in self.k.vehicle.get_arrived_ids()
                for key in states.keys()}
        # # TODO(@ev) figure out why done is not being set
        # if 'av_0' in done and done['av_0']:
        #     self.done_list.extend(['av_0'])
        no_avs_left = len([veh_id for veh_id in self.k.vehicle.get_ids() if ('av' in veh_id or 'rl' in veh_id)]) == 0
        # it can take a little bit for all the AVs to enter the system
        if veh_crash or ped_crash or (no_avs_left and self.time_counter > 50 / self.sim_step) \
                or self.time_counter >= self.env_params.horizon:
            # import ipdb; ipdb.set_trace()
            done['__all__'] = True
        else:
            done['__all__'] = False

        # done_ids = [veh_id for veh_id in self.k.vehicle.get_arrived_ids() if ('av' in veh_id or 'rl' in veh_id
        #                                                                        or veh_id in self.observed_rl_ids)]
        arrived_ids = self.k.vehicle.get_arrived_ids()
        ids_to_check = copy(self.k.vehicle.get_ids())
        ids_to_check += [veh_id for veh_id in self.k.vehicle.get_arrived_ids()]
        done_ids = [veh_id for veh_id in ids_to_check if ((('av' in veh_id or 'rl' in veh_id
                                                                              or veh_id in self.observed_rl_ids) and
                                                                      (veh_id in arrived_ids or
                                                                      veh_id in collide_ids)) and
                                                                      veh_id not in self.done_ids)]

        done_ids = [done_id for done_id in done_ids if 'human' not in done_id]
        # vehicles might not have exited so if done all is true, we need to return a state for
        # every vehicle currently in the system that hasn't recieved a done yet
        if done["__all__"]:
            done_ids += [veh_id for veh_id in self.k.vehicle.get_rl_ids() if (veh_id not in self.done_ids
                                                                              and veh_id not in done_ids)]

        # if crash:
        #     import ipdb; ipdb.set_trace()

        for rl_id in done_ids:
            done[rl_id] = True
            if rl_id in collide_ids:
                self.reward[rl_id] = self.env_params.additional_params["ped_collision_penalty"]
                reward[rl_id] = self.env_params.additional_params["ped_collision_penalty"]
            else:
                self.exit_time[rl_id] = self.time_counter * self.sim_step
                # the episode ended because of a crash so we probably shouldn't be getting a postiive
                # reward for it
                if done['__all__']:
                    self.reward[rl_id] = 0.0
                    reward[rl_id] = 0.0
                else:
                    self.reward[rl_id] = 1.0
                    reward[rl_id] = 1.0

            states[rl_id] = -1 * np.ones(self.observation_space.shape[0])
            self.done_ids.update([rl_id])

        return states, reward, done, infos

    def reset(self, new_inflow_rate=None):
        """Reset the environment.

        This method is performed in between rollouts. It resets the state of
        the environment, and re-initializes the vehicles in their starting
        positions.

        If "shuffle" is set to True in InitialConfig, the initial positions of
        vehicles is recalculated and the vehicles are shuffled.

        Returns
        -------
        observation : dict of array_like
            the initial observation of the space. The initial reward is assumed
            to be zero.
        """
        # print(self.ped_transition_cnt)
        self.controller_dict = {}
        self.time_counter = 0
        # last time we saw a vehicle
        self.last_seen = 0
        self.reward = {}
        self.exit_time = {}
        self.done_ids = set()
        self.prior_probs = -1 * np.ones(len(self.prior_names))
        self.prev_loc_ped_state = {loc: 0 for loc in range(NUM_PED_LOCATIONS)}
        # dict to store the counts for each possible transiion
        self.ped_transition_cnt = {loc: {'00': 1, '01': 1, '10': 1, '11': 1} for loc in range(NUM_PED_LOCATIONS)}
        self.observed_rl_ids = set()

        # Now that we've passed the possibly fake init steps some rl libraries
        # do, we can feel free to actually render things
        if self.should_render:
            self.sim_params.render = True
            # got to restart the simulation to make it actually display anything
            # self.restart_simulation(self.sim_params)

        # reset the time counter
        self.time_counter = 0

        self.arrival_order = {}
        self.inside_intersection = set()
        self.got_to_intersection = set()
        self.done_list = []

        # warn about not using restart_instance when using inflows
        if len(self.net_params.inflows.get()) > 0 and \
                not self.sim_params.restart_instance:
            print(
                "**********************************************************\n"
                "**********************************************************\n"
                "**********************************************************\n"
                "WARNING: Inflows will cause computational performance to\n"
                "significantly decrease after large number of rollouts. In \n"
                "order to avoid this, set SumoParams(restart_instance=True).\n"
                "**********************************************************\n"
                "**********************************************************\n"
                "**********************************************************"
            )

        if self.sim_params.restart_instance or \
                (self.step_counter > 2e6 and self.simulator != 'aimsun'):
            self.step_counter = 0
            # issue a random seed to induce randomness into the next rollout
            self.sim_params.seed = np.random.randint(0, 1e5)

            self.k.vehicle = deepcopy(self.initial_vehicles)
            self.k.vehicle.master_kernel = self.k
            # restart the sumo instance
            self.restart_simulation(self.sim_params)

        # perform shuffling (if requested)
        elif self.initial_config.shuffle:
            self.setup_initial_state()

        # clear all vehicles from the network and the vehicles class
        if self.simulator == 'traci':
            for veh_id in self.k.kernel_api.vehicle.getIDList():  # FIXME: hack
                try:
                    self.k.vehicle.remove(veh_id)
                except (FatalTraCIError, TraCIException):
                    print(traceback.format_exc())

        # clear all vehicles from the network and the vehicles class
        # FIXME (ev, ak) this is weird and shouldn't be necessary
        for veh_id in list(self.k.vehicle.get_ids()):
            # do not try to remove the vehicles from the network in the first
            # step after initializing the network, as there will be no vehicles
            if self.step_counter == 0:
                continue
            try:
                self.k.vehicle.remove(veh_id)
            except (FatalTraCIError, TraCIException):
                print("Error during start: {}".format(traceback.format_exc()))

        randomize_vehicles = self.env_params.additional_params.get("randomize_vehicles", False)

        # TODO(@ev) enable
        # if randomize_vehicles:
        #     num_rl, num_human = 0, 0
        #     rl_index = np.random.randint(len(self.initial_ids))
        #     for i in range(len(self.initial_ids)):
        #         veh_id = self.initial_ids[i]
        #         type_id, edge, lane_index, pos, speed, depart_time = \
        #             self.initial_state[veh_id]
        #         if self.net_params.additional_params["randomize_routes"]:
        #             if i == rl_index:
        #                 type_id = 'rl'
        #             else:
        #                 type_id = np.random.choice(['rl', 'human'])
        #
        #         if type_id == 'rl':
        #             veh_name = 'rl_' + str(num_rl)
        #             type_id = veh_name
        #             num_rl += 1
        #         else:
        #             veh_name = 'human_' + str(num_human)
        #             type_id = veh_name
        #             num_human += 1
        #
        #         try:
        #             if type_id == 'rl_1' or veh_name == 'rl_1':
        #                 import ipdb; ipdb.set_trace()
        #             self.k.vehicle.add(
        #                 veh_id=veh_name,
        #                 type_id=type_id,
        #                 edge=edge,
        #                 lane=lane_index,
        #                 pos=pos,
        #                 speed=speed,
        #                 depart_time=depart_time)
        #
        #
        #         except (FatalTraCIError, TraCIException):
        #             # if a vehicle was not removed in the first attempt, remove it
        #             # now and then reintroduce it
        #             self.k.vehicle.remove(veh_name)
        #             if self.simulator == 'traci':
        #                 self.k.kernel_api.vehicle.remove(veh_name)  # FIXME: hack
        #             self.k.vehicle.add(
        #                 veh_id=veh_name,
        #                 type_id=type_id,
        #                 edge=edge,
        #                 lane=lane_index,
        #                 pos=pos,
        #                 speed=speed,
        #                 depart_time=depart_time)
        #
        # else:
        for veh_id in self.initial_ids:
            type_id, edge, lane_index, pos, speed, depart_time = \
                self.initial_state[veh_id]
            try:
                self.k.vehicle.add(
                    veh_id=veh_id,
                    type_id=type_id,
                    edge=edge,
                    lane=lane_index,
                    pos=pos,
                    speed=speed,
                    depart_time=depart_time)
            except (FatalTraCIError, TraCIException):
                # if a vehicle was not removed in the first attempt, remove it
                # now and then reintroduce it
                self.k.vehicle.remove(veh_id)
                if self.simulator == 'traci':
                    self.k.kernel_api.vehicle.remove(veh_id)  # FIXME: hack
                self.k.vehicle.add(
                    veh_id=veh_id,
                    type_id=type_id,
                    edge=edge,
                    lane=lane_index,
                    pos=pos,
                    speed=speed,
                    depart_time=depart_time)

        # set the past intersection rewarded set to empty
        self.past_intersection_rewarded_set = set()
        # set the near intersection rewarded set to empty
        self.near_intersection_rewarded_set_1 = set()
        # set the near intersection rewarded set to empty
        self.near_intersection_rewarded_set_2 = set()
        # set the near intersection rewarded set to empty
        self.near_intersection_rewarded_set_3 = set()

        # advance the simulation in the simulator by one step
        self.k.simulation.simulation_step()

        # update the information in each kernel to match the current state
        self.k.update(reset=True)

        # update the colors of vehicles
        if self.sim_params.render:
            self.k.vehicle.update_vehicle_colors()

        # check to make sure all vehicles have been spawned
        if len(self.initial_ids) > self.k.vehicle.num_vehicles:
            missing_vehicles = list(
                set(self.initial_ids) - set(self.k.vehicle.get_ids()))
            msg = '\nNot enough vehicles have spawned! Bad start?\n' \
                  'Missing vehicles / initial state:\n'
            for veh_id in missing_vehicles:
                msg += '- {}: {}\n'.format(veh_id, self.initial_state[veh_id])
            raise FatalFlowError(msg=msg)

        # perform (optional) warm-up steps before training
        for _ in range(self.env_params.warmup_steps):
            observation, _, _, _ = self.step(rl_actions=None)

        # render a frame
        self.render(reset=True)

        return self.get_state()

    def update_curriculum(self, training_iter):
        if training_iter > 30:
            self.speed_reward_coefficient = 0
        else:
            self.speed_reward_coefficient = 1

    def find_visible_objects(self, veh_id, radius):
        """For a given vehicle ID, find the IDs of all the objects that are within a radius of them

        Parameters
        ----------
        veh_id : str
            The id of the vehicle whose visible objects we want to compute
        radius : float
            How large of a circle we want to search around

        Returns
        -------
        visible_vehicles, visible_pedestrians, visible_l : [str], [str], [int]
            Returns three lists of the IDs of vehicles, IDs of pedestrians, IDs of crosswalk lanes
            that are within a radius of the car and are unobscured.

        2 types of crosswalk lanes: 
            a) zebra crossings: these contain a 'w'
            b) intersection ped walkways: these contain a 'c'
        """
        visible_vehicles, visible_pedestrians, visible_lanes = self.k.vehicle.get_viewable_objects(veh_id, self.k.pedestrian, self.k.network.kernel_api.lane, radius)

        return visible_vehicles, visible_pedestrians, visible_lanes

    def get_action_mask(self, valid_agent):
        """If a valid agent, return a 0 in the position of the no-op action. If not, return a 1 in that position
        and a zero everywhere else."""
        if valid_agent:
            temp_list = np.array([1 for _ in range(self.action_space.n)])
            temp_list[0] = 0
        else:
            temp_list = np.array([0 for _ in range(self.action_space.n)])
            temp_list[0] = 1
        return temp_list

    ############################
    # get_state() helper methods #
    ############################

    def update_intersection_state(self, rl_id):
        curr_edge = self.k.vehicle.get_edge(rl_id)
        if curr_edge in self.edge_to_int:
            if rl_id in self.inside_intersection and self.k.vehicle.get_position(rl_id) > 4:
                self.inside_intersection.remove(rl_id)
        else:
            self.inside_intersection.add(rl_id)
            self.got_to_intersection.add(rl_id)

    def convert_edge_to_int(self, edge):
        if edge in self.edge_to_int:
            curr_edge = self.edge_to_int[edge]
        else:
            curr_edge = -1
        return curr_edge

    def get_self_obs(self, rl_id, visible_peds, visible_lanes):
        """For a given vehicle ID, get the observation info related explicitly to the given vehicle itself
        
        Parameters
        ----------
        veh_id: str
            vehicle id
        visible_peds: [ped_obj, ped_obj, ...]
            list of pedestrian objects visible to vehicle id

        Returns
        -------
        observation : [int / float]
            list of integer / float values [yaw, speed, turn_num, edge_pos, ped_1, ..., ped_4]
            ped_i is a binary value indicating whether (1) or not (0) 
            there's a pedestrian in grid cell i of veh in question
        
        """
        veh_x, veh_y = self.k.vehicle.get_orientation(rl_id)[:2]
        yaw = self.k.vehicle.get_yaw(rl_id)
        speed = self.k.vehicle.get_speed(rl_id)
        curr_edge = self.convert_edge_to_int(self.k.vehicle.get_edge(rl_id))
        edge_pos = self.k.vehicle.get_position(rl_id)
        if self.k.vehicle.get_edge(rl_id) in self.in_edges:
            edge_pos = 50 - edge_pos
        start, end = self.k.vehicle.get_route(rl_id)
        start = self.edge_to_int[start]
        end = self.edge_to_int[end]
        turn_num = (end - start) % 8
        if turn_num == 1:
            turn_num = 0  # turn right
        elif turn_num == 3:
            turn_num = 1  # go straight
        else:
            turn_num = 2  # turn left
        # subtract by one since we're not including the pedestrian here
        observation = [yaw / 360, speed / 20, turn_num / 2, curr_edge / 8, end / 8, edge_pos / 50,
                       veh_x / 300.0, veh_y / 300.0, self.arrival_position(rl_id) / 5.0]
        if self.use_grid:
            ped_param = self.get_grid_ped_params(visible_peds, rl_id)
        else:
            ped_param = self.four_way_ped_params(visible_peds, visible_lanes, ground_truth=False)
        observation.extend(ped_param)
        return observation

    def arrived_before(self, veh_1, veh_2):
        """Return 1 if vehicle veh_1 arrived at the intersection before vehicle veh_2. Else, return 0."""
        if veh_1 not in self.arrival_order:
            return 1
        elif self.arrival_order[veh_1] < self.arrival_order[veh_2]:
            return 1
        else:
            return 0

    def arrival_position(self, veh_1):
        """Return arrival position if vehicle has arrived. Else, return -1"""
        if veh_1 not in self.arrival_order:
            return -2
        else:
            return self.arrival_order[veh_1]

    def curr_ped_state(self):
        """Return a list containing the ground truth state of pedestrians wrt the four locations. 
        Idx i of the list corresponds to location i"""
        locs = [0] * NUM_PED_LOCATIONS

        ped_kernel = self.k.pedestrian
        for loc in range(NUM_PED_LOCATIONS):
            for ped_id in ped_kernel.get_ids():
                ped_edge = ped_kernel.get_edge(ped_id)
                loc = self.edge_to_loc(ped_edge, ped_id)
                if loc is not None:
                    locs[loc] = 1

        return locs

    def get_non_ped_obs(self, veh_id):
        """Return all the obs for vehicle veh_id aside from the updated probabilities for 
        the four crosswalk pedestrian params
        
        Returns
        -------
        non_ped_obs : list
            self_obs_names + [None, None, None, None] + veh_obs_names x (max_num_objects - 1)
        """
        max_objects = self.env_params.additional_params["max_num_objects"]
        num_self_no_ped_obs = len(self.self_obs_names)
        num_other_no_ped_obs = len(self.veh_obs_names)
        non_ped_obs = np.zeros(num_self_no_ped_obs + NUM_PED_LOCATIONS + num_other_no_ped_obs * max_objects)

        visible_vehicles, visible_pedestrians = self.find_visible_objects(veh_id, self.search_radius)

        # sort visible vehicles by angle where 0 degrees starts facing the right side of the vehicle
        visible_vehicles = sorted(visible_vehicles, key=lambda v: \
                (self.k.vehicle.get_relative_angle(veh_id, \
                self.k.vehicle.get_orientation(v)[:2]) + 90) % 360)

        # TODO(@nliu)add get x y as something that we store from TraCI (no magic numbers)
        non_ped_obs[:num_self_no_ped_obs] = self.get_self_obs(veh_id, visible_pedestrians)[:num_self_no_ped_obs]
        veh_x, veh_y = self.k.vehicle.get_orientation(veh_id)[:2]

        # setting the 'arrival' order feature: 1 is if agent arrives before; 0 if agent arrives after
        for idx, obs_veh_id in enumerate(visible_vehicles):
            
            before = self.arrival_position(veh_id)

            observed_yaw = self.k.vehicle.get_yaw(obs_veh_id)
            observed_speed = self.k.vehicle.get_speed(obs_veh_id)
            observed_x, observed_y = self.k.vehicle.get_orientation(obs_veh_id)[:2]
            rel_x = observed_x - veh_x
            rel_y = observed_y - veh_y

            # Consider the first 3 visible vehicles
            # TODO(@evinitsky) magic numbers
            if idx <= 2:
                non_ped_obs[(idx * num_other_no_ped_obs) + num_self_no_ped_obs: \
                            num_other_no_ped_obs * (idx + 1) + num_self_no_ped_obs] = \
                            [observed_yaw, observed_speed, rel_x, rel_y, before]

        return non_ped_obs

    def four_way_ped_params(self, visible_pedestrians, visible_lanes, ground_truth=True, veh_id="1"):
        """For a given RL agent's visible pedestrians and visible lanes, return a
        length 4 ternary indicator array for the 'source' car's pedestrian visibility state vector.
        If using ground truth, return a length 4 binary indicator array of whether there are pedestrians
        on each of the crosswalks, regardless of the actual source vehicle.
        
        Parameters
        ----------
        visible_pedestrians : list[str]
            list of all pedestrian id's visible to the source rl car
        visible_lanes: list[str]
            list of all lane id's visible to the source rl car
        ground_truth: boolean
            flag for whether we should return ground truth pedestrian-crosswalk states

        Returns
        -------
        cross_walk: list[int]
            a length NUM_PED_LOCATIONS list denoting pedestrian visibility w.r.t. a source car,
            or the ground truth for pedestrian-on-crosswalk state

        Definitions
        -----------
        SUMO has 3 types of pedestrian 'edges':
            i) sidewalk - these are the straight sideways, formatted as (1.1)--(0.1)
            ii) walkway - these are the diagonal parts inside the intersection, formatted as "(1.1)_w[num]"
            iii) crossing - these are the stripey zebra crossings, formatted as "(1.1)_c[num]" 

        Let's define a 'crosswalk' as follows (check google slides)

        For a stripey zebra crossing (location 0), crosswalk 0 would be the regions with the stars

                 |  |    |    |  |
                 |  |    |    |  |        
                 |  |    |    |  |
                 |  |    |    |  |        
                 |  |    |    |  |
                 |**|    |    |**|   
        _________|**||*||*||*||**|
        _________       
        
        _________

        _________
        _________
    
 
        We define three values for flags:
            1 = the car physically sees a pedestrian on that crosswalk
            0 = the car physically sees no pedestrians on that crosswalk
            -1 = the car can't tell if there's a pedestrian on that crosswalk (because crosswalk is not fully in view)
        
        | Crossing number | Sumo Edge location   |
        | --------------- | -------------------- |
        | 0               | c0, w1_0             |
        | 1               | c1, w2_0             |
        | 2               | c2, w3_0             |
        | 3               | c3, w0_0             |
        """

        # ped_kernel = self.k.pedestrian
        # # import ipdb; ipdb.set_trace()
        # if ground_truth:
        #     locs = self.curr_ped_state()

        # else:
        #     locs = [-1] * NUM_PED_LOCATIONS
        #     lane_visible_arr = [[0,0] for _ in range(NUM_PED_LOCATIONS)]

        #     for lane in visible_lanes:
        #         # check if a lane is fully in view (i.e. need both)
        #         junction = lane.split("_")[0][1:]
        #         if JUNCTION_ID == junction:
        #             loc = self.edge_to_loc(lane)
        #             if loc is not None:
        #                 if 'c' in lane:
        #                     lane_visible_arr[loc][0] = 1
        #                 elif 'w' in lane:
        #                     lane_visible_arr[loc][1] = 1

        #     for idx, val in enumerate(lane_visible_arr):
        #         if val[0] == val[1] == 1:
        #             locs[idx] = 0

        #     for ped_id in visible_pedestrians:
        #         ped_edge = ped_kernel.get_edge(ped_id)
        #         loc = self.edge_to_loc(ped_edge, ped_id)
        #         if loc is not None:
        #             locs[loc] = 1

        # return locs

        cross_walk = [0] * NUM_PED_LOCATIONS
        ped_kernel = self.k.pedestrian

        if ground_truth:
            pedestrians = ped_kernel.get_ids()
            for ped in pedestrians:  # pedestrians = all pedestrians in the simulation
                for cw in range(NUM_PED_LOCATIONS):
                    if self.is_ped_on_cross_walk(ped, cw):
                        cross_walk[cw] = 1
                # print(ped_kernel.get_edge(ped))
            # print(cross_walk)
        else:
            for ped in visible_pedestrians:  # pedestrians = all pedestrians in the simulation visible to source vehicle
                for cw in range(NUM_PED_LOCATIONS):
                    if self.is_ped_on_cross_walk(ped, cw):
                        cross_walk[cw] = 1

            # check why we can’t see a ped on cross_walk[cw]: is it because the ‘source’ vehicle can’t see the entirety of the cross_walk?
            # In that case, the pedestrian’s obscured and we set cross_walk[cw] = -1. Else, we keep cross_walk[cw] = 0.

            # for cw in range(NUM_PED_LOCATIONS):
            #     if cross_walk[cw] == 0:
            #         if not self.is_cross_walk_visible(veh_id, cw):
            #             cross_walk[cw] = -1
        # print(cross_walk)
        return cross_walk

    def is_ped_on_cross_walk(self, ped_id, cw):
        """Check if there's a pedestrian ped_id on crosswalk cw

        crossing cw: i) zebra crossing "c" or waiting walkway platform "w"
                     ii) <= 1m on sidewalk edge away from walkway platform

        Parameters
        ----------
        ped_id: str
            ped_id is the pedestrian we care about
        cw: int
            relevant crosswalk's number
        
        Returns
        -------
        True if ped_id is on cw, False if ped_id isn't on cw
        """
        ped_kernel = self.k.pedestrian
        # check :(1.1)_w2, (1.0)--(1.1), :(1.1)_c2
        cw_edges = self.get_cross_walk_edge_names(cw)
        ped_edge = ped_kernel.get_edge(ped_id)

        if ped_edge in cw_edges:
            if "c" in ped_edge or "w" in ped_edge:
                return True
            elif "-" in ped_edge:
                edge_num = self.edge_to_num[ped_edge]
                edge_pos = ped_kernel.get_lane_position(ped_id)
                return (edge_num % 2 == 0 and edge_pos >= 47) or (edge_num % 2 == 1 and edge_pos <= 1)

        else:
            return False

    def get_cross_walk_edge_names(self, cw):
        """Return the names of edges corresponding to crosswalk cw
        
        crossing cw: i) zebra crossing "c" or waiting walkway platform "w"
                     ii) <= 1m on sidewalk edge away from walkway platform

                     OR

                     cross_walk i: ci, corner_i, corner_{i+1} (mod 4)
                     and, corner_i: w_i, edge_{2i}, edge_{2i - 1} (mod 8)

        Parameters
        ----------
        cw: int
            number of the crosswalk in question
        
        Returns
        -------
        cw_names: list[str]
            string of all edges related to crosswalk cw

        NB, format of edge string names are as follows
        
        ":(1.1)_w2", "(1.0)--(1.1)", ":(1.1)_c2"
        """

        cw_names = []
        c = ":" + JUNCTION_ID + "_c" + str(cw)
        cw_names.append(c)

        for j in range(2):
            i = (cw + j) % 4
            w = ":" + JUNCTION_ID + "_w" + str(i)
            edge_1 = self.num_to_edge[2 * i]
            edge_2 = self.num_to_edge[(2 * i - 1) % 8]
            cw_names.extend([w, edge_1, edge_2])

        return cw_names

    def is_cross_walk_visible(self, veh_id, cw):
        pass

    # @DeprecationWarning
    def edge_to_loc(self, lane, ped_id=None):
        """Map a lane id to its corresponding corresponds value. 

        Parameters
        -----------
        lane: str
            a SUMO lane
        ped_id: str
            id of a pedestrian

        Return
        ------
        int
            integer corresponding to a physical location
        None if pedestrian isn't on one of the physical locations

        Also, to overcome the issue of a pedestrian being really far away along a walkway
        that it might as well be irrelevant, I'll check if the pedestrian is also within a 
        certain radius of the corresponding crosswalk.
        """
        # check if the lane is within the intersection area / crosswalk
        if "c" not in lane and "w" not in lane:
            return None
        else:
            # c corresponds to something inside an intersection area in SUMO
            if 'c' in lane:
                lane = lane.split("_")[1]
                return int(lane[1])
            # w corresponds to something else inside an intersection area in SUMO
            if 'w' in lane:
                # Ugly code alert - there's a lot of random formatting to get things done in SUMO
                if ped_id:
                    # check if pedestrian is in an appropriate radius 
                    lane_kernel = self.k.network.kernel_api.lane
                    lane_id_copy = list(lane)
                    w_idx = lane.index('w')
                    lane_id_copy[w_idx] = 'c'
                    lane_id_copy[-1] = str((int(lane_id_copy[-1]) - 1) % 4)
                    corresp_crosswalk_id = "".join(lane_id_copy)
                    pts = lane_kernel.getShape(corresp_crosswalk_id + '_0')
                    pt_a, pt_b = pts[0], pts[1]
                    cross_walk_center = ((pt_a[0] + pt_b[0]) / 2, (pt_a[1] + pt_b[1]) / 2)
                    walkway_length = lane_kernel.getLength(lane + '_0')
                    lane = lane.split("_")[1]
                    if self.in_circle_radius(cross_walk_center, walkway_length * 1.3,
                                             self.k.pedestrian.get_position(ped_id)):
                        return (int(lane[1]) - 1) % 4
                    else:
                        return None
                else:
                    lane = lane.split("_")[1]
                    return (int(lane[1]) - 1) % 4

    def in_circle_radius(self, center, radius, pt):
        """Return True if pt is within the circle specified by the center point and the radius"""
        dist_to_center = np.sqrt((center[0] - pt[0]) ** 2 + (center[1] - pt[1]) ** 2)
        return dist_to_center <= radius

    def get_grid_ped_params(self, visible_pedestrians, rl_id):
        veh_x, veh_y = self.k.vehicle.get_orientation(rl_id)[:2]
        ped_param = [0, 0, 0, 0, 0, 0]
        # we assuming there's only 1 ped?
        if len(visible_pedestrians) > 0:
            ped_x, ped_y = self.k.pedestrian.get_position(visible_pedestrians[0])
            rel_x = ped_x - veh_x
            rel_y = ped_y - veh_y
            rel_angle = self.k.vehicle.get_relative_angle(rl_id, (ped_x, ped_y))
            rel_angle = (rel_angle + 90) % 360
            dist = math.sqrt((rel_x ** 2) + (rel_y ** 2))
            if rel_angle < 60:
                if dist < 15:
                    ped_param[0] = 1
                else:
                    ped_param[1] = 1
            elif rel_angle < 120:
                if dist < 15:
                    ped_param[2] = 1
                else:
                    ped_param[3] = 1
            elif rel_angle < 180:
                if dist < 15:
                    ped_param[4] = 1
                else:
                    ped_param[5] = 1
            else:
                raise RuntimeError("Relative Angle is Invalid")
        return ped_param
