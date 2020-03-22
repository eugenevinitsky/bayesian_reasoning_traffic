"""Environment testing scenario one of the bayesian envs."""
import numpy as np
from gym.spaces.box import Box
from flow.core.rewards import desired_velocity
from flow.envs.multiagent.base import MultiEnv

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
    "search_radius": 50
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

        self.rl_set = set()

    @property
    def observation_space(self):
        """See class definition."""
        max_objects = self.env_params.additional_params["max_num_objects"]
        # the items per object are relative X, relative Y, speed, whether it is a pedestrian, and its yaw TODO(@nliu no magic 5 number)
        return Box(-float('inf'), float('inf'), shape=(7 + max_objects * len(self.observation_names),), dtype=np.float32)

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
        """For a radius around the car, return the 3 closest objects with their X, Y position relative to you,
        their speed, a flag indicating if they are a pedestrian or not, and their yaw."""

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

        for rl_id in self.rl_set:
            if rl_id in self.k.vehicle.get_arrived_ids():
                obs.update({rl_id: np.zeros(self.observation_space.shape[0])})

        for rl_id in self.k.vehicle.get_rl_ids():
            self.rl_set.add(rl_id)

            num_obs = len(self.observation_names)

            observation = np.zeros(self.observation_space.shape[0])   #TODO(KL) Check if this makes sense
            #TODO(@nliu): currently not using pedestrians
            visible_vehicles, visible_pedestrians = self.find_visible_objects(rl_id, self.search_radius)

            # sort visible vehicles by angle where 0 degrees starts facing the right side of the vehicle
            visible_vehicles = sorted(visible_vehicles, key=lambda v: \
                    (self.k.vehicle.get_relative_angle(rl_id, \
                    self.k.vehicle.get_orientation(v)[:2]) + 90) % 360)

            # TODO(@nliu)add get x y as something that we store from TraCI (no magic numbers)
            veh_x, veh_y = self.k.vehicle.get_orientation(rl_id)[:2]
            yaw = self.k.vehicle.get_yaw(rl_id)
            speed = self.k.vehicle.get_speed(rl_id)
            edge = self.k.vehicle.get_edge(rl_id)
            edge_int = edge_to_int.get(edge, -1)
            edge_pos = self.k.vehicle.get_position(rl_id)

            observation[:4] = [yaw, speed, edge_int, edge_pos]

            '''
            pedestrian_in_view = 0
            if len(visible_pedestrians) > 0:
                pedestrian_in_view = 1

            observation[4] = pedestrian_in_view
            '''
            if len(visible_pedestrians) > 0:
                ped_x, ped_y = self.k.pedestrian.get_position(visible_pedestrians[0])
                ped_orientation = self.k.pedestrian.get_yaw(visible_pedestrians[0])
                rel_x = ped_x - veh_x
                rel_y = ped_y - veh_y
                observation[4:7] = [rel_x, rel_y, ped_orientation]
            else:
                observation[4:7] = [0, 0, 0]

            #TODO(@nliu) sort by angle
            for index, veh_id in enumerate(visible_vehicles):
                observed_yaw = self.k.vehicle.get_yaw(veh_id)
                observed_speed = self.k.vehicle.get_speed(veh_id)
                observed_x, observed_y = self.k.vehicle.get_orientation(veh_id)[:2]
                rel_x = observed_x - veh_x
                rel_y = observed_y - veh_y

                if index <= 2: 
                    observation[(index * 4) + 7: 4 * (index + 1) + 7] = \
                            [observed_yaw, observed_speed, rel_x, rel_y]

            #print(observation)
            obs.update({rl_id: observation})
        
        return obs

    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        # in the warmup steps
        if rl_actions is None:
            return {}

        rewards = {}

        for rl_id in self.rl_set:
            if rl_id in self.k.vehicle.get_arrived_ids():
                rewards[rl_id] = 0

        for rl_id in self.k.vehicle.get_rl_ids():


            # TODO(@evinitsky) pick the right reward
            reward = 0

            collision_vehicles = self.k.simulation.get_collision_vehicle_ids()
            collision_pedestrians = self.k.vehicle.get_pedestrian_crash(rl_id, self.k.pedestrian)

            if len(collision_pedestrians) > 0:
                reward = -50
            elif rl_id in collision_vehicles:
                reward = -30
            else:
                # TODO(@nliu & evinitsky) positive reward?
                # reward = rl_actions[rl_id][0] / 10 # small reward for going forward
                reward = self.k.vehicle.get_speed(rl_id) / 10.0 #speed may be better for no braking randomly
                # reward = -1

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
        visible_vehicles, visible_pedestrians = self.k.vehicle.get_viewable_objects(
                veh_id,
                self.k.pedestrian,
                radius)

        return visible_vehicles, visible_pedestrians
