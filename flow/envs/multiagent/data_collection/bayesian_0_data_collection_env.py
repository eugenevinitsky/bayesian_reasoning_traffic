"""Environment testing scenario one of the bayesian envs."""
import numpy as np
from gym.spaces.box import Box
from flow.core.rewards import desired_velocity
from flow.envs.multiagent.base import MultiEnv
from flow.envs.ring.accel import AccelEnv
# TODO(KL) means KL's reminder for KL

ADDITIONAL_ENV_PARAMS = {
    # maximum acceleration of autonomous vehicles
    'max_accel': 1,
    # maximum deceleration of autonomous vehicles
    'max_decel': 1,
    # desired velocity for all vehicles in the network, in m/s
    "target_velocity": 25,
    # how many objects in our local radius we want to return
    "max_num_objects": 3,
    # how large of a radius to search in for a given vehicle in meters
    "search_radius": 20,
    # specifies whether vehicles are to be sorted by position during a
    # simulation step. If set to True, the environment parameter
    # self.sorted_ids will return a list of all vehicles sorted in accordance
    # with the environments 
    'sort_vehicles': False
}
# TODO(KL) Not 100 sure what sorted_vehicles means 


class Bayesian0DataCollectionEnv(AccelEnv):
    """

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

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        for p in ADDITIONAL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))

        super().__init__(env_params, sim_params, network, simulator)
        self.observation_names = ["rel_x", "rel_y", "speed", "yaw"]
        self.search_radius = self.env_params.additional_params["search_radius"]

    @property
    def observation_space(self):
        """See class definition."""
        max_objects = self.env_params.additional_params["max_num_objects"]
        # the items per object are relative X, relative Y, speed, whether it is a pedestrian, and its yaw TODO(@nliu no magic 5 number)
        return Box(-float('inf'), float('inf'), shape=(5 + max_objects * len(self.observation_names),), dtype=np.float32)

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
            for rl_id, actions in rl_actions.items():
                accel = actions[0]
                self.k.vehicle.apply_acceleration(rl_id, accel)

    def get_state(self):
        """For a radius around the only car, return the car's own state information i.e. speed, yaw, edge, distance on edge 
        they are a pedestrian or not, and their yaw."""
        obs = {}

        edge_to_int = {
                "(1.1)--(2.1)" : 0,
                "(2.1)--(1.1)" : 1,
                "(1.1)--(1.2)" : 2,
                "(1.2)--(1.1)" : 3,
                "(1.1)--(0.1)" : 4,
                "(0.1)--(1.1)" : 5,
                "(1.1)--(1.0)" : 6,
                "(1.0)--(1.1)" : 7
        }

        for rl_id in self.k.vehicle.get_ids():
            # TODO(@nliu)add get x y as something that we store from TraCI (no magic numbers)
            num_obs = len(self.observation_names)
            observation = np.zeros(self.observation_space.shape[0])   #TODO(KL) Check if this makes sense
            #TODO(@nliu): currently not using pedestrians
            visible_vehicles, visible_pedestrians = self.find_visible_objects(rl_id, self.search_radius)
            # sort visible vehicles by angle where 0 degrees starts facing the right side of the vehicle
            visible_vehicles = sorted(visible_vehicles, key=lambda v: \
                    (self.k.vehicle.get_relative_angle(rl_id, \
                    self.k.vehicle.get_orientation(v)[:2]) + 90) % 360)

            veh_x, veh_y = self.k.vehicle.get_orientation(rl_id)[:2]
            yaw = self.k.vehicle.get_yaw(rl_id)
            speed = self.k.vehicle.get_speed(rl_id)
            edge = self.k.vehicle.get_edge(rl_id)
            edge_int = edge_to_int.get(edge, -1)
            edge_pos = self.k.vehicle.get_position(rl_id)

            observation[:4] = [yaw, speed, edge_int, edge_pos]
            
            observation[4] = 0 # TODO(@nliu) pedestrians implementation later


            # #TODO(@nliu) sort by angle
            # for index, veh_id in enumerate(visible_vehicles):
            #     observed_yaw = self.k.vehicle.get_yaw(veh_id)
            #     observed_speed = self.k.vehicle.get_speed(veh_id)
            #     observed_x, observed_y = self.k.vehicle.get_orientation(veh_id)[:2]
            #     rel_x = observed_x - veh_x
            #     rel_y = observed_y - veh_y

            #     if index <= 2: 
            #         observation[(index * 4) + 5: 4 * (index + 1) + 5] = \
            #                 [observed_yaw, observed_speed, rel_x, rel_y]

            # obs.update({rl_id: observation})
        
        return obs

    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        # in the warmup steps
        if rl_actions is None:
            return {}

        rewards = {}
        for rl_id in self.k.vehicle.get_rl_ids():

            # TODO(@evinitsky) pick the right reward

            reward = 0

            # TODO(@nliu) verify this works
            collision_vehicles = self.k.simulation.get_collision_vehicle_ids()
            if rl_id in collision_vehicles:
                reward = -10
            else:
                # TODO(@nliu & evinitsky) positive reward?
                reward = rl_actions[rl_id][0] / 10 # small reward for going forward

            rewards[rl_id] = reward

        return rewards


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
        visible_vehicles, visible_pedestrians = self.k.vehicle.get_viewable_objects( \
                veh_id,
                self.k.pedestrian,
                radius)

        return visible_vehicles, visible_pedestrians

    def additional_command(self):
        """See parent class.

        Define which vehicles are observed for visualization purposes, and
        update the sorting of vehicles using the self.sorted_ids variable.
        """
        # specify observed vehicles
        if self.k.vehicle.num_rl_vehicles > 0:
            for veh_id in self.k.vehicle.get_human_ids():
                self.k.vehicle.set_observed(veh_id)

        # update the "absolute_position" variable
        for veh_id in self.k.vehicle.get_ids():
            this_pos = self.k.vehicle.get_x_by_id(veh_id)

            if this_pos == -1001:
                # in case the vehicle isn't in the network
                self.absolute_position[veh_id] = -1001
            else:
                change = this_pos - self.prev_pos.get(veh_id, this_pos)
                self.absolute_position[veh_id] = \
                    (self.absolute_position.get(veh_id, this_pos) + change) \
                    % self.k.network.length()
                self.prev_pos[veh_id] = this_pos

    @property
    def sorted_ids(self):
        """Sort the vehicle ids of vehicles in the network by position.

        This environment does this by sorting vehicles by their absolute
        position, defined as their initial position plus distance traveled.

        Returns
        -------
        list of str
            a list of all vehicle IDs sorted by position
        """
        if self.env_params.additional_params['sort_vehicles']:
            return sorted(self.k.vehicle.get_ids(), key=self._get_abs_position)
        else:
            return self.k.vehicle.get_ids()

    def _get_abs_position(self, veh_id):
        """Return the absolute position of a vehicle."""
        return self.absolute_position.get(veh_id, -1001)

    def reset(self):
        """See parent class.

        This also includes updating the initial absolute position and previous
        position.
        """
        obs = super().reset()

        for veh_id in self.k.vehicle.get_ids():
            self.absolute_position[veh_id] = self.k.vehicle.get_x_by_id(veh_id)
            self.prev_pos[veh_id] = self.k.vehicle.get_x_by_id(veh_id)

        return obs