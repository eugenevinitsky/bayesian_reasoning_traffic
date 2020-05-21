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
    # how large of a radius to search vehicles in for a given vehicle in meters
    "search_veh_radius": 50,
    # # how large of a radius to search pedestrians in for a given vehicle in meters
    "search_ped_radius": 22,
    # whether we use an observation space configured for MADDPG
    "maddpg": False
}

HARD_BRAKE_PENALTY = 0.001
NUM_PED_LOCATIONS = 4
JUNCTION_ID = '(1.1)'

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
        for p in ADDITIONAL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))

        super().__init__(env_params, sim_params, network, simulator)

        self.veh_obs_names = ["rel_x", "rel_y", "speed", "yaw", "arrive_before"]
        self.self_obs_names = ["yaw", "speed", "turn_num", "curr_edge", "edge_pos", "ped_in_0", "ped_in_1", "ped_in_2", "ped_in_3"]
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

        self.near_intersection_rewarded_set_1 = set()
        self.near_intersection_rewarded_set_2 = set()
        self.near_intersection_rewarded_set_3 = set()

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
        obs_space = Box(-float('inf'), float('inf'), shape=(len(self.self_obs_names) + max_objects * len(self.veh_obs_names),), dtype=np.float32)
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
        intersection_length = self.net_params.additional_params['grid_array']['inner_length']
        dist_to_intersection = intersection_length - self.k.vehicle.get_position(veh_id) 
        return not (self.k.vehicle.get_edge(veh_id) == self.k.vehicle.get_route(veh_id)[0] and \
                dist_to_intersection > 20)
    
    def past_intersection(self, veh_id):
        """Return True if vehicle is at least 20m past the intersection (we had control back to SUMO at this point) & false if not""" #TODO(KL)
        on_post_intersection_edge = self.k.vehicle.get_edge(veh_id) == self.k.vehicle.get_route(veh_id)[-1]       
        if on_post_intersection_edge and self.k.vehicle.get_position(veh_id) > 8: # vehicle arrived at final destination, 8 is a random distance
            return True
        elif self.k.vehicle.get_edge(veh_id) == '':
            return True
        return False

    def get_state(self):
        """For a radius around the car, return the 3 closest objects with their X, Y position relative to you,
        their speed, a flag indicating if they are a pedestrian or not, and their yaw."""
        obs = {}
        num_self_obs = len(self.self_obs_names)
        num_veh_obs = len(self.veh_obs_names)
        for veh_id in self.k.vehicle.get_ids():
            if veh_id not in self.arrival_order and self.arrived_intersection(veh_id):
                self.arrival_order[veh_id] = len(self.arrival_order)

        for rl_id in self.k.vehicle.get_rl_ids():
            if rl_id in self.past_intersection_rewarded_set:
                continue
            if self.arrived_intersection(rl_id):
                self.rl_set.add(rl_id)
                assert rl_id in self.arrival_order

                # MADDPG hack
                if isinstance(self.observation_space, Dict):
                    observation = np.zeros(self.observation_space["obs"].shape[0])
                else:
                    observation = np.zeros(self.observation_space.shape[0])   #TODO(KL) Check if this makes sense

                visible_vehicles, visible_pedestrians, visible_lanes = self.find_visible_objects(rl_id, self.search_veh_radius)

                # sort visible vehicles by angle where 0 degrees starts facing the right side of the vehicle
                visible_vehicles = sorted(visible_vehicles, key=lambda v: \
                        (self.k.vehicle.get_relative_angle(rl_id, \
                        self.k.vehicle.get_orientation(v)[:2]) + 90) % 360)

                # TODO(@nliu)add get x y as something that we store from TraCI (no magic numbers)
                observation[:num_self_obs] = self.get_self_obs(veh_id, visible_pedestrians, visible_lanes)
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
                        observation[(index * num_veh_obs) + num_self_obs: num_veh_obs * (index + 1) + num_self_obs] = \
                                [observed_yaw / 360, observed_speed / 20, 
                                        rel_x / 50, rel_y / 50, before / 5]

                obs.update({rl_id: observation})
        return obs

    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        # in the warmup steps
        if rl_actions is None:
            return {}

        rewards = {}
        for rl_id in self.k.vehicle.get_rl_ids():            
            if self.past_intersection(rl_id):
                if rl_id in self.past_intersection_rewarded_set:
                    continue
                else:
                    rewards[rl_id] = 25 / 100
                    self.past_intersection_rewarded_set.add(rl_id)
                    continue
                continue

            if self.arrived_intersection(rl_id) and not self.past_intersection(rl_id):
                reward = 0
                edge_pos = self.k.vehicle.get_position(rl_id)
                if 38 < edge_pos < 40:
                    if rl_id in self.near_intersection_rewarded_set_1:
                        pass
                    else:
                        reward = 200
                        self.near_intersection_rewarded_set_1.add(rl_id)

                if 44 < edge_pos < 46:
                    if rl_id in self.near_intersection_rewarded_set_1:
                        pass
                    else:
                        reward = 250
                        self.near_intersection_rewarded_set_1.add(rl_id)
            
                if 47 < edge_pos < 50:
                    if rl_id in self.near_intersection_rewarded_set_3:
                        pass
                    else:
                        reward = 300
                        self.near_intersection_rewarded_set_3.add(rl_id)

                # TODO(@evinitsky) pick the right reward
                collision_vehicles = self.k.simulation.get_collision_vehicle_ids()
                collision_pedestrians = self.k.vehicle.get_pedestrian_crash(rl_id, self.k.pedestrian)
                inside_intersection = rl_id in self.inside_intersection
                if len(collision_pedestrians) > 0:
                    reward = -800
                elif rl_id in collision_vehicles:
                    reward = -1000
                # else:
                #     reward += self.k.vehicle.get_speed(rl_id) / 100.0 * self.speed_reward_coefficient
                    # TODO(@nliu & evinitsky) positive reward?
                    # reward = rl_actions[rl_id][0] / 10 # small reward for going forward
                # if rl_id in self.inside_intersection:
                #     # TODO(KL) 'hard-brake' as negative acceleration?
                #     if self.k.vehicle.get_acceleration(rl_id) < -0.8:
                #         reward -= HARD_BRAKE_PENALTY

                rewards[rl_id] = reward / 100

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

        randomize_ped = True
    

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
        visible_vehicles, visible_pedestrians, visible_lanes = self.k.vehicle.get_viewable_objects(
                veh_id,
                self.k.pedestrian,
                self.k.network.kernel_api.lane,
                radius)
        
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
        observation = []
        veh_x, veh_y = self.k.vehicle.get_orientation(rl_id)[:2]
        yaw = self.k.vehicle.get_yaw(rl_id)
        speed = self.k.vehicle.get_speed(rl_id)

        curr_edge = self.k.vehicle.get_edge(rl_id)
        if curr_edge in self.edge_to_int:
            curr_edge = self.edge_to_int[curr_edge]
            if rl_id in self.inside_intersection: 
                self.inside_intersection.remove(rl_id)
        else:
            curr_edge = -1
            self.inside_intersection.add(rl_id)
        edge_pos = self.k.vehicle.get_position(rl_id)
        # import ipdb; ipdb.set_trace()
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
        # subtract by one since we're not including the pedestrian here
        observation[:len(self.self_obs_names) - 1] = [yaw / 360, speed / 20, turn_num / 2, curr_edge / 8, edge_pos / 50]
        ped_param = self.ped_params(visible_peds, visible_lanes)
        observation.extend(ped_param)
        return observation

    def arrived_before(self, veh_1, veh_2):
        """Return 1 if vehicle veh_1 arrived at the intersection before vehicle veh_2. Else, return 0."""
        if veh_2 not in self.arrival_order:
            return 1
        elif self.arrival_order[veh_1] < self.arrival_order[veh_2]:
            return 1
        else:
            return 0

    def ped_params(self, visible_pedestrians, visible_lanes):
        """Return length 4 ternary indicator array for an RL car's pedestrian visibility state vector.
        
        1 = the car physically sees a pedestrian on that crosswalk
        0 = the car physically sees no pedestrians on that crosswalk
        -1 = the car can't tell if there's a pedestrian on that crosswalk (because crosswalk is not fully in view)

        The 4 possible physical locations are indicated in the slide (?) currently in my journal.     
        **Simplifying Assumption** pedestrians only traverse crossings counterclockwise:
        
        | Crossing number | Sumo Edge location   |
        | --------------- | -------------------- |
        | 0               | c0, w1_0             |
        | 1               | c1, w2_0             |
        | 2               | c2, w3_0             |
        | 3               | c3, w0_0             |
        """
        locs = [-1] * NUM_PED_LOCATIONS
        lane_visible_arr = [[0,0] for _ in range(NUM_PED_LOCATIONS)]
        ped_kernel = self.k.pedestrian
        for lane in visible_lanes:
            # check if a lane is fully in view (i.e. need both)
            junction = lane.split("_")[0][1:]
            if JUNCTION_ID == junction:
                loc = self.edge_to_loc(lane)
                if loc is not None:
                    if 'c' in lane:
                        lane_visible_arr[loc][0] = 1
                    elif 'w' in lane:
                        lane_visible_arr[loc][1] = 1

        for idx, val in enumerate(lane_visible_arr):
            if val[0] == val[1] == 1:
                locs[idx] = 0

        for ped_id in visible_pedestrians:
            ped_edge = ped_kernel.get_edge(ped_id)
            loc = self.edge_to_loc(ped_edge, ped_id)
            if loc is not None:
                locs[loc] = 1
        return locs

    def edge_to_loc(self, lane, ped_id=None):
        """Return the number that a lane or pedestrian's physical location corresponds to. 
        Return None if pedestrian isn't on one of the physical locations

        Also, to overcome the issue of a pedestrian being really far away along a walkway
        that it might as well be irrelevant, I'll check if the pedestrian is also within a 
        certain radius of the corresponding crosswalk.
        """
        if "c" not in lane and "w" not in lane:
            return None
        else:
            if 'c' in lane:
                lane = lane.split("_")[1]    
                return int(lane[1])
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
                    if self.in_circle_radius(cross_walk_center, walkway_length * 1.3, self.k.pedestrian.get_position(ped_id)):
                        return (int(lane[1]) - 1) % 4
                    else:
                        return None
                else:
                    lane = lane.split("_")[1]    
                    return (int(lane[1]) - 1) % 4
    
    def in_circle_radius(self, center, radius, pt):
        """Return True if pt is within the circle specified by the center point and the radius"""
        dist_to_center = np.sqrt((center[0] - pt[0])**2 + (center[1] - pt[1])**2)
        return dist_to_center <= radius

            
            



