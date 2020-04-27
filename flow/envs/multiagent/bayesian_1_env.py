"""Environment testing scenario one of the bayesian envs."""
from copy import deepcopy
import math
import numpy as np
from gym.spaces import Box, Dict, Discrete
from flow.core.rewards import desired_velocity
from flow.envs.multiagent.base import MultiEnv

from traci.exceptions import FatalTraCIError
from traci.exceptions import TraCIException
from flow.utils.exceptions import FatalFlowError
# from bayesian_inference.bayesian_inference_PPO import create_black_box

# TODO(KL) means KL's reminder for KL

ADDITIONAL_ENV_PARAMS = {
    # maximum acceleration of autonomous vehicles
    'max_accel': 4.5,
    # maximum deceleration of autonomous vehicles
    'max_decel': -2.6,
    # desired velocity for all vehicles in the network, in m/s
    "target_velocity": 25,
    # how many objects in our local radius we want to return
    "max_num_objects": 3,
    # how large of a radius to search in for a given vehicle in meters
    "search_radius": 50,
    # whether we use an observation space configured for MADDPG
    "maddpg": False
}


class Bayesian1Env(MultiEnv):
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
        for p in ADDITIONAL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))

        super().__init__(env_params, sim_params, network, simulator)

        # wonder if it's better to specify the file path or the kind of policy (the latter?)
        
        # self.updated_probs_fn = create_black_box("PPO")
        self.observation_names = ["rel_x", "rel_y", "speed", "yaw", "arrive_before"]
        self.search_radius = self.env_params.additional_params["search_radius"]
        self.maddpg = self.env_params.additional_params["maddpg"]
        if self.maddpg:
            self.max_num_agents = 3
            self.num_actions = 5
            self.action_values = np.linspace(start=-np.abs(self.env_params.additional_params['max_decel']),
                                             stop=self.env_params.additional_params['max_accel'], num=self.num_actions)
            # self.default_state = {idx: {"obs": np.zeros(self.observation_space.spaces['obs'].shape[0]),
            #                             "action_mask": self.get_action_mask(valid_agent=False)}
            #                       for idx in range(self.max_num_agents)}
            self.default_state = {idx: -1 * np.ones(self.observation_space.shape[0])
                                  for idx in range(self.max_num_agents)}
        # variable to encourage vehicle to move in curriculum training
        self.speed_reward_coefficient = 1
        # track all rl_vehicles: hack to compute the last reward of an rl vehicle (reward for arriving, set states to 0)
        self.rl_set = set()
        # feature for arrival
        self.arrival_order = {}

        # TODO hardcoding
        # this is used for maddpg
        if self.maddpg:
            self.idx_to_av_id = {i: 'rl_{}'.format(i) for i in range(self.max_num_agents)}
            self.av_id_to_idx = {'rl_{}'.format(i): i for i in range(self.max_num_agents)}

        self.edge_to_int = {
                "(1.1)--(2.1)" : 0,
                "(2.1)--(1.1)" : 1,
                "(1.1)--(1.2)" : 2,
                "(1.2)--(1.1)" : 3,
                "(1.1)--(0.1)" : 4,
                "(0.1)--(1.1)" : 5,
                "(1.1)--(1.0)" : 6,
                "(1.0)--(1.1)" : 7
        }

        self.in_edges = ["(2.1)--(1.1)",
                "(1.2)--(1.1)",
                "(0.1)--(1.1)",
                "(1.0)--(1.1)"]

    @property
    def observation_space(self):
        """See class definition."""
        max_objects = self.env_params.additional_params["max_num_objects"]
        # the items per object are relative X, relative Y, speed, whether it is a pedestrian, and its yaw TODO(@nliu no magic 5 number)
        obs_space = Box(-float('inf'), float('inf'), shape=(11 + max_objects * len(self.observation_names),), dtype=np.float32)
        if self.maddpg:
            # TODO(@evinitsky) put back the action mask
            # return Dict({"obs": obs_space, "action_mask": Box(0, 1, shape=(self.action_space.n,))})
            return obs_space
        else:
            return obs_space

    @property
    def action_space(self):
        """See class definition."""
        return Box(
            low=-np.abs(self.env_params.additional_params['max_decel']),
            high=self.env_params.additional_params['max_accel'],
            shape=(1,),  # (4,),
            dtype=np.float32)

    def _apply_rl_actions(self, rl_actions):
        """See class definition."""
        # in the warmup steps, rl_actions is None
        if rl_actions:
            # if self.maddpg:
            #     accel_list = []
            #     rl_ids = []
            #     for rl_id, action in rl_actions.items():
            #         # 0 is the no-op
            #         import ipdb; ipdb.set_trace()
            #         if action > 0:
            #             accel = self.action_values[action]
            #             accel_list.append(accel)
            #             rl_ids.append(self.idx_to_av_id[rl_id])
            #     self.k.vehicle.apply_acceleration(rl_ids, accel_list)
            # else:
            rl_ids = []
            accels = []
            for rl_id, actions in rl_actions.items():
                if not self.arrived_intersection(rl_id):
                    continue
                if rl_id in self.k.vehicle.get_rl_ids():
                    self.k.vehicle.set_speed_mode(rl_id, 'aggressive')
                accel = actions[0]
                rl_ids.append(rl_id)
                accels.append(accel)
            self.k.vehicle.apply_acceleration(rl_ids, accels)

    def arrived_intersection(self, veh_id):
        """Return True if vehicle is at or past the intersection and false if not."""
        # When vehicle exits, route is []
        if len(self.k.vehicle.get_route(veh_id)) == 0: # vehicle arrived to final destination
            return True
        return not (self.k.vehicle.get_edge(veh_id) == self.k.vehicle.get_route(veh_id)[0] and \
                self.k.vehicle.get_position(veh_id) < 49)
                
    def get_state(self):
        """For a radius around the car, return the 3 closest objects with their X, Y position relative to you,
        their speed, a flag indicating if they are a pedestrian or not, and their yaw."""
        obs = {}

        # TODO(KL) MADDPG hack
        for rl_id in self.rl_set:
            if rl_id in self.k.vehicle.get_arrived_ids():
                if isinstance(self.observation_space, Dict):
                    temp_dict = {}
                    for k, space in self.observation_space.spaces.items():
                        if isinstance(space, Discrete):
                            temp_dict.update({k: 0})
                        else:
                            temp_dict.update({k: np.zeros(space.shape[0])})

                else:
                    obs.update({rl_id: np.zeros(self.observation_space.shape[0])})

        for veh_id in self.k.vehicle.get_ids():
            if veh_id not in self.arrival_order and self.arrived_intersection(veh_id):
                self.arrival_order[veh_id] = len(self.arrival_order)

        for rl_id in self.k.vehicle.get_rl_ids():
            if self.arrived_intersection(rl_id):
                self.rl_set.add(rl_id)

                assert rl_id in self.arrival_order

                # MADDPG hack
                if isinstance(self.observation_space, Dict):
                    observation = np.zeros(self.observation_space["obs"].shape[0])
                else:
                    observation = np.zeros(self.observation_space.shape[0])   #TODO(KL) Check if this makes sense
                visible_vehicles, visible_pedestrians = self.find_visible_objects(rl_id, self.search_radius)

                # sort visible vehicles by angle where 0 degrees starts facing the right side of the vehicle
                visible_vehicles = sorted(visible_vehicles, key=lambda v: \
                        (self.k.vehicle.get_relative_angle(rl_id, \
                        self.k.vehicle.get_orientation(v)[:2]) + 90) % 360)

                # TODO(@nliu)add get x y as something that we store from TraCI (no magic numbers)
                observation[:11] = self.get_self_obs(veh_id, visible_pedestrians)
                veh_x, veh_y = self.k.vehicle.get_orientation(rl_id)[:2]

                # setting the 'arrival' order feature: 1 is if agent arrives before; 0 if agent arrives after
                for index, veh_id in enumerate(visible_vehicles):

                    before = self.arrived_before(rl_id, veh_id)

                    observed_yaw = self.k.vehicle.get_yaw(veh_id)
                    observed_speed = self.k.vehicle.get_speed(veh_id)
                    observed_x, observed_y = self.k.vehicle.get_orientation(veh_id)[:2]
                    rel_x = observed_x - veh_x
                    rel_y = observed_y - veh_y

                    # Consider the first 3 visible vehicles
                    if index <= 2:
                        observation[(index * 5) + 11: 5 * (index + 1) + 11] = \
                                [observed_yaw / 360, observed_speed / 20, 
                                        rel_x / 50, rel_y / 50, before / 5]
                    if max(observation) > 1 or min(observation) < -1:
                        print(observation)

                    # append updated prob of ped in grid i to state vector
                    # self 

                obs.update({rl_id: observation})

        if self.maddpg and len(self.rl_set) > 0:

            # TODO(@evinitsky) think this doesn't have to be a deepcopy
            veh_info_copy = deepcopy(self.default_state)
            # veh_info_copy.update({self.av_id_to_idx[rl_id]: {"obs": obs[rl_id],
            #                                   "action_mask": self.get_action_mask(valid_agent=True)}
            #                       for rl_id in obs.keys()})
            veh_info_copy.update({self.av_id_to_idx[rl_id]: obs[rl_id] for rl_id in obs.keys()})
            obs = veh_info_copy

        return obs

    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        # in the warmup steps
        if rl_actions is None:
            return {}

        rewards = {}

        for rl_id in self.k.vehicle.get_rl_ids():
            if self.arrived_intersection(rl_id):
                # TODO(@evinitsky) pick the right reward
                reward = 0

                collision_vehicles = self.k.simulation.get_collision_vehicle_ids()
                collision_pedestrians = self.k.vehicle.get_pedestrian_crash(rl_id, self.k.pedestrian)

                if len(collision_pedestrians) > 0:
                    reward = -300
                elif rl_id in collision_vehicles:
                    reward = -100
                else:
                    reward = self.k.vehicle.get_speed(rl_id) / 100.0 * self.speed_reward_coefficient
                    '''
                    if self.k.vehicle.get_edge(rl_id) != self.k.vehicle.get_route(rl_id)[0]:
                        if rl_actions[rl_id] < 0:
                            reward += rl_actions[rl_id][0] / 10
                    '''
                    # TODO(@nliu & evinitsky) positive reward?
                    # reward = rl_actions[rl_id][0] / 10 # small reward for going forward

                rewards[rl_id] = reward / 100

        for rl_id in self.rl_set:
            '''
            if self.arrived_intersection(rl_id):
                if rl_id in self.k.vehicle.get_arrived_ids():
                    rewards[rl_id] = 50 / 100
            '''
            if rl_id in self.k.vehicle.get_arrived_ids():
                rewards[rl_id] = 25 / 100

        if self.maddpg:
            if len(self.rl_set) > 0:
                temp_rewards = {self.av_id_to_idx[rl_id]: 0 for rl_id in self.av_id_to_idx.keys()}
                temp_rewards.update({self.av_id_to_idx[rl_id]: reward for rl_id, reward in rewards.items()})
                rewards = temp_rewards

        return rewards


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
	    # Now that we've passed the possibly fake init steps some rl libraries
        # do, we can feel free to actually render things
        if self.should_render:
            self.sim_params.render = True
            # got to restart the simulation to make it actually display anything
            # self.restart_simulation(self.sim_params)

        # reset the time counter
        self.time_counter = 0

        self.arrival_order = {}

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

        # reintroduce the initial vehicles to the network # TODO(KL) I've set randomize_drivers to false - need to subclass
        randomize_drivers = False
        if randomize_drivers:
            num_rl, num_human = 0, 0
            rl_index = np.random.randint(len(self.initial_ids))
            for i in range(len(self.initial_ids)):
                veh_id = self.initial_ids[i]
                type_id, edge, lane_index, pos, speed, depart_time = \
                    self.initial_state[veh_id]
                if self.net_params.additional_params.get("randomize_routes", False):
                    if i == rl_index:
                        type_id = 'rl'
                    else:
                        type_id = np.random.choice(['rl', 'human'])

                if type_id == 'rl':
                    veh_name = 'rl_' + str(num_rl)
                    num_rl += 1
                else:
                    veh_name = 'human_' + str(num_human)
                    num_human += 1

                try:
                    self.k.vehicle.add(
                        veh_id=veh_name,
                        type_id=type_id,
                        edge=edge,
                        lane=lane_index,
                        pos=pos,
                        speed=speed,
                        depart_time=depart_time)
                        

                except (FatalTraCIError, TraCIException):
                    # if a vehicle was not removed in the first attempt, remove it
                    # now and then reintroduce it
                    self.k.vehicle.remove(veh_name)
                    if self.simulator == 'traci':
                        self.k.kernel_api.vehicle.remove(veh_name)  # FIXME: hack
                    self.k.vehicle.add(
                        veh_id=veh_name,
                        type_id=type_id,
                        edge=edge,
                        lane=lane_index,
                        pos=pos,
                        speed=speed,
                        depart_time=depart_time)

        else:
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
        visible_vehicles, visible_pedestrians : [str], [str]
            Returns two lists of the IDs of vehicles and pedestrians that are within a radius of the car and are unobscured

        """
        visible_vehicles, visible_pedestrians = self.k.vehicle.get_viewable_objects(
                veh_id,
                self.k.pedestrian,
                radius)

        return visible_vehicles, visible_pedestrians

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

    def get_self_obs(self, rl_id, visible_peds):
        """For a given vehicle ID, get the observation info related explicitly to the given vehicle itself
        
        Parameters
        ----------
        veh_id: str
            vehicle id
        visible_peds: [ped_obj, ped_obj, ...]
            list of pedestrian objects visible to vehicle id
        edge_to_int:
            ...
        in_edges:
            ...

        Returns
        -------
        observation : [int / float]
            list of integer / float values [yaw, speed, turn_num, edge_pos, ped_1, ..., ped_6]
            ped_i is a binary value indicating whether (1) or not (0) 
            there's a pedestrian in grid cell i of veh in question
        
        """
        observation = []
        veh_x, veh_y = self.k.vehicle.get_orientation(rl_id)[:2]
        yaw = self.k.vehicle.get_yaw(rl_id)
        speed = self.k.vehicle.get_speed(rl_id)
        curr_edge = self.k.vehicle.get_edge(rl_id)
        if curr_edge in self.edge_to_int:
            curr_edge = self.edge_to_int[curr_edge]
        else:
            curr_edge = -1
        edge_pos = self.k.vehicle.get_position(rl_id)
        if self.k.vehicle.get_edge(rl_id) in self.in_edges:
            edge_pos = 50 - edge_pos
        start, end = self.k.vehicle.get_route(rl_id)
        start = self.edge_to_int[start]
        end = self.edge_to_int[end]
        turn_num = (end - start) % 8
        if turn_num == 1:
            turn_num = 0 # turn right
        elif turn_num == 3:
            turn_num = 1 # go straight
        else:
            turn_num = 2 # turn left

        observation[:5] = [yaw / 360, speed / 20, turn_num / 2, curr_edge / 8, edge_pos / 50]

        ped_param = [0, 0, 0, 0, 0, 0]
        # we assuming there's only 1 ped?
        if len(visible_peds) > 0:
            ped_x, ped_y = self.k.pedestrian.get_position(visible_peds[0])
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
        observation[5:11] = ped_param

        return observation

    def arrived_before(self, veh_1, veh_2):
        """Return 1 if vehicle veh_1 arrived at the intersection before vehicle veh_2. Else, return 0."""
        if veh_2 not in self.arrival_order:
            return 1
        elif self.arrival_order[veh_1] < self.arrival_order[veh_2]:
            return 1
        else:
            return 0