"""Contains a list of custom velocity controllers."""

# TODO(@evinitsky) move to util
# from flow.utils.rllib import create_agent_from_path
from flow.core.params import SumoCarFollowingParams
from flow.controllers.base_controller import BaseController
from bayesian_inference.inference import get_filtered_posteriors
import numpy as np


class PreTrainedController(BaseController):
    def __init__(self,
                 veh_id,
                 car_following_params,
                 path,
                 checkpoint_num):
        BaseController.__init__(
            self, veh_id, car_following_params, delay=1.0)
        self.path = path
        self.agent = create_agent_from_path(path, checkpoint_num)

    def get_accel(self, env):
        state = env.get_state()
        if not env.past_intersection(self.veh_id) and len(state) > 0 and self.veh_id in state.keys():
            action = self.agent.compute_action(state[self.veh_id], policy_id='av', explore=False)
            # action = env.discrete_actions_to_accels[np.argmax(q_val)]
            return action
        else:
            return None

    def get_discrete_action(self, env):
        state = env.get_state()
        if len(state) > 0 and self.veh_id in state.keys():
            action = self.agent.compute_action(state[self.veh_id], policy_id='av')
            return action
        else:
            return None

    def get_action(self, env, allow_junction_control=False):
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


class FullStop(BaseController):
    def __init__(self,
                 veh_id,
                 car_following_params):
        """Instantiate FollowerStopper."""
        BaseController.__init__(
            self, veh_id, car_following_params, delay=1.0)

    def get_accel(self, env):
        return -4.5


class RuleBasedIntersectionController(BaseController):
    def __init__(self,
                 veh_id,
                 car_following_params,
                 noise=0.0):
        BaseController.__init__(
            self, veh_id, car_following_params, noise=noise)

        self.edge_to_num = {
            "(1.2)--(1.1)" : 0,
            "(1.1)--(1.2)" : 0,
            "(2.1)--(1.1)" : 1,
            "(1.1)--(2.1)" : 1,
            "(1.0)--(1.1)" : 2,
            "(1.1)--(1.0)" : 2,
            "(0.1)--(1.1)" : 3,
            "(1.1)--(0.1)" : 3
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

    def get_action_with_ped(self, env, state, ped=None, change_speed_mode=True, always_return_action=False):
        """Compute the action given the state. Lets us pass in modified states.

        always_return_action: bool
            If true, we return an action even if we have no yet 'arrived' at the intersection
        """

        # ped_pos = [i + len(env.self_obs_names) for i in range (4)]

        if ped:
            visible_peds = ped
        else:
            visible_peds = env.four_way_ped_params(env.k.pedestrian.get_ids(), [], ground_truth=True)

        desired_pos = 48
        # if env.k.vehicle.get_position(self.veh_id) < desired_pos and (env.k.vehicle.get_edge(self.veh_id) == env.k.vehicle.get_route(self.veh_id)[0]):
        #     action = 2.6
        #     lead_vel = 0
        #     this_vel = env.k.vehicle.get_speed(self.veh_id)
        #
        #     h = desired_pos - env.k.vehicle.get_position(self.veh_id)
        #     dv = lead_vel - this_vel
        #
        #     safe_velocity = 2 * h / env.sim_step + dv - this_vel * (2 * self.delay)
        #
        #     this_vel = env.k.vehicle.get_speed(self.veh_id)
        #     sim_step = env.sim_step
        #
        #     if this_vel + action * sim_step > safe_velocity:
        #         if safe_velocity > 0:
        #             return (safe_velocity - this_vel) / sim_step
        #         # hard brake to not overshoot
        #         else:
        #             return -this_vel / sim_step
        #     else:
        #         return action
        # elif env.k.vehicle.get_edge(self.veh_id) == env.k.vehicle.get_route(self.veh_id)[-1]:
        #     return None
        #
        # print(env.k.vehicle.get_position(self.veh_id))

        # we sometimes query this as a dummy controller, we don't want it to change the speed mode at that point
        if not always_return_action and env.k.vehicle.get_position(self.veh_id) < desired_pos and (
                env.k.vehicle.get_edge(self.veh_id) == env.k.vehicle.get_route(self.veh_id)[0]):
            if change_speed_mode:
                env.k.vehicle.set_speed_mode(self.veh_id, 'right_of_way')
            # print('position is ', env.k.vehicle.get_position(self.veh_id))
            return None
        else:
            if change_speed_mode:
                env.k.vehicle.set_speed_mode(self.veh_id, 'aggressive')

        start_edge, end_edge = env.k.vehicle.get_route(self.veh_id)
        start, end = self.edge_to_num[start_edge], self.edge_to_num[end_edge]

        # if you're past the start edge, you shouldn't stop just because there's a vehicle on the edge behind you
        if (visible_peds[start] and env.k.vehicle.get_edge(self.veh_id) == start_edge)\
                or visible_peds[end] and not env.past_intersection(self.veh_id) \
                and self.veh_id not in env.inside_intersection:
            return -4.5

        # inch forward if no vehicle is before you in the order, otherwise go
        # arrival_order = [env.arrival_order[veh_id] for veh_id in env.arrival_order
        #                  if veh_id in env.k.vehicle.get_ids() and
        #                  (env.k.vehicle.get_edge(veh_id) == env.k.vehicle.get_route(veh_id)[0]
        #                  or veh_id in env.inside_intersection)
        #                  and not env.past_intersection(veh_id)]
        arrival_order = [env.arrival_order[veh_id] for veh_id in env.arrival_order
                         if veh_id in env.k.vehicle.get_ids() and not env.past_intersection(veh_id)]

        arrival_order_dict = {veh_id: env.arrival_order[veh_id] for veh_id in env.arrival_order
                              if veh_id in env.k.vehicle.get_ids() and not env.past_intersection(veh_id)}

        # if env.k.vehicle.get_edge(self.veh_id) == env.k.vehicle.get_route(self.veh_id)[-1] and env.k.vehicle.get_speed(self.veh_id) < 0.01:
        #     import ipdb; ipdb.set_trace()
        # if len(arrival_order) == 0 or env.past_intersection(self.veh_id):
        if env.past_intersection(self.veh_id):
            if change_speed_mode:
                env.k.vehicle.set_speed_mode(self.veh_id, 'right_of_way')
            return 2.6
        else:
            start, end = env.k.vehicle.get_route(self.veh_id)
            start = self.edge_to_int[start]
            end = self.edge_to_int[end]
            turn_num = (end - start) % 8
            # right turn is always safe unless someone else is also going to end on this edge
            if turn_num == 1:
                final_edge = [env.k.vehicle.get_route(veh_id)[-1] == env.k.vehicle.get_route(self.veh_id)[-1]
                              for veh_id in arrival_order_dict.keys()
                              if veh_id != self.veh_id]
                if not np.any(final_edge):
                    return 2.6
            if self.veh_id not in env.arrival_order or env.arrival_order[self.veh_id] == np.min(arrival_order):
                return 2.6
            else:
                return -4.5

    def get_accel(self, env):
        """Drive up to the intersection. Go if there are no pedestrians and you're first in the arrival order"""
        state = env.state_for_id(self.veh_id)
        return self.get_action_with_ped(env, state)

    def get_action(self, env, allow_junction_control=False):
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


class RuleBasedInferenceController(RuleBasedIntersectionController):
    def __init__(self,
                 veh_id,
                 car_following_params,
                 noise=0.0,
                 inference_noise=0.0):
        RuleBasedIntersectionController.__init__(
            self, veh_id, car_following_params, noise=noise)

        self.edge_to_num = {
            "(1.2)--(1.1)" : 0,
            "(1.1)--(1.2)" : 0,
            "(2.1)--(1.1)" : 1,
            "(1.1)--(2.1)" : 1,
            "(1.0)--(1.1)" : 2,
            "(1.1)--(1.0)" : 2,
            "(0.1)--(1.1)" : 3,
            "(1.1)--(0.1)" : 3
        }
        # we use this to track a RuleBasedController for every other controller in the scene
        # since the L1 controllers assume everything else is a L0 controller
        self.controller_dict = {}
        self.priors = {}
        self.accel = None
        self.inference_noise = inference_noise

    def get_accel(self, env):
        """Drive up to the intersection. Go if there are no pedestrians and you're first in the arrival order"""
        visible_vehicles, visible_pedestrians, visible_lanes = env.k.vehicle.get_viewable_objects(
            self.veh_id,
            env.k.pedestrian,
            env.k.network.kernel_api.lane,
            env.search_veh_radius)
        ped_vals = env.four_way_ped_params(visible_pedestrians, visible_lanes, ground_truth=False)
        state = env.state_for_id(self.veh_id)
        action = self.get_action_with_ped(env, state, ped=ped_vals)

        for veh_id in visible_vehicles:
            if veh_id not in self.controller_dict and veh_id != self.veh_id:
                self.controller_dict[veh_id] = RuleBasedIntersectionController(veh_id,
                                                                               car_following_params=SumoCarFollowingParams(
                                                                                   min_gap=2.5,
                                                                                   decel=7.5,
                                                                                   # avoid collisions at emergency stops
                                                                                   speed_mode="right_of_way",
                                                                                    )
                                                                               )
            # don't do inference on yourself lol
            if veh_id != self.veh_id and not env.past_intersection(veh_id):
                dummy_obs = np.zeros(env.observation_space.shape[0])
                _, visible_pedestrians, visible_lanes = env.find_visible_objects(veh_id,
                                                                                  env.search_veh_radius)
                ped_params = env.four_way_ped_params(visible_pedestrians, visible_lanes)
                # TODO fix magic numbers
                dummy_obs[[10, 11, 12, 13]] = ped_params
                # import ipdb; ipdb.set_trace()
                controller = env.k.vehicle.get_acc_controller(veh_id)
                if hasattr(controller, 'accel'):
                    acceleration = env.k.vehicle.get_acc_controller(veh_id).accel
                else:
                    acceleration = env.k.vehicle.get_acc_controller(veh_id).get_accel(env)
                # import ipdb; ipdb.set_trace()
                # we pass a zero of states because it's just a dummy obs, only the ped part of it affects the behavior
                # import ipdb; ipdb.set_trace()
                updated_ped_probs, self.priors[veh_id] = get_filtered_posteriors(env, self.controller_dict[veh_id], acceleration,
                                                                                 np.zeros(env.observation_space.shape[0]),
                                                                                 self.priors.get(veh_id,
                                                                                                 {}),
                                                                                 veh_id,
                                                                                 noise_std=self.inference_noise)
                # note that I got this backwards, this is actually the probability of no peds, which
                # is why this is a less than
                # if hasattr(env, 'query_env'):
                #     updated_ped_probs, _ = get_filtered_posteriors(env, self.controller_dict[veh_id], acceleration,
                #                             np.zeros(env.observation_space.shape[0]),
                #                             self.priors.get(veh_id,
                #                                             {}),
                #                             veh_id)
                # the second condition is just so videos don't look stupid
                if np.any(np.array(updated_ped_probs) > 0.8) and env.k.vehicle.get_position(self.veh_id) > 45.0:
                    # we use this to check if we got it correctly. We should uh, automate this.
                    # TODO(@evinitsky) automate this
                    if not hasattr(env, 'query_env'):
                        if env.time_counter < 20.0:
                            print('acceleration is', acceleration)
                            print(updated_ped_probs)
                    get_filtered_posteriors(env, self.controller_dict[veh_id], acceleration,
                                            np.zeros(env.observation_space.shape[0]),
                                            self.priors.get(veh_id,
                                                            {}),
                                            veh_id)
                    action = -4.5

                self.accel = action

        return action



class FollowerStopper(BaseController):
    """Inspired by Dan Work's... work.

    Dissipation of stop-and-go waves via control of autonomous vehicles:
    Field experiments https://arxiv.org/abs/1705.01693

    Usage
    -----
    See base class for example.

    Parameters
    ----------
    veh_id : str
        unique vehicle identifier
    v_des : float, optional
        desired speed of the vehicles (m/s)
    """

    def __init__(self,
                 veh_id,
                 car_following_params,
                 v_des=15,
                 danger_edges=None):
        """Instantiate FollowerStopper."""
        BaseController.__init__(
            self, veh_id, car_following_params, delay=1.0,
            fail_safe='safe_velocity')

        # desired speed of the vehicle
        self.v_des = v_des

        # maximum achievable acceleration by the vehicle
        self.max_accel = car_following_params.controller_params['accel']

        # other parameters
        self.dx_1_0 = 4.5
        self.dx_2_0 = 5.25
        self.dx_3_0 = 6.0
        self.d_1 = 1.5
        self.d_2 = 1.0
        self.d_3 = 0.5
        self.danger_edges = danger_edges if danger_edges else {}

    def find_intersection_dist(self, env):
        """Find distance to intersection.

        Parameters
        ----------
        env : flow.envs.Env
            see flow/envs/base.py

        Returns
        -------
        float
            distance from the vehicle's current position to the position of the
            node it is heading toward.
        """
        edge_id = env.k.vehicle.get_edge(self.veh_id)
        # FIXME this might not be the best way of handling this
        if edge_id == "":
            return -10
        if 'center' in edge_id:
            return 0
        edge_len = env.k.network.edge_length(edge_id)
        relative_pos = env.k.vehicle.get_position(self.veh_id)
        dist = edge_len - relative_pos
        return dist

    def get_accel(self, env):
        """See parent class."""
        lead_id = env.k.vehicle.get_leader(self.veh_id)
        this_vel = env.k.vehicle.get_speed(self.veh_id)
        lead_vel = env.k.vehicle.get_speed(lead_id)

        if self.v_des is None:
            return None

        if lead_id is None:
            v_cmd = self.v_des
        else:
            dx = env.k.vehicle.get_headway(self.veh_id)
            dv_minus = min(lead_vel - this_vel, 0)

            dx_1 = self.dx_1_0 + 1 / (2 * self.d_1) * dv_minus**2
            dx_2 = self.dx_2_0 + 1 / (2 * self.d_2) * dv_minus**2
            dx_3 = self.dx_3_0 + 1 / (2 * self.d_3) * dv_minus**2
            v = min(max(lead_vel, 0), self.v_des)
            # compute the desired velocity
            if dx <= dx_1:
                v_cmd = 0
            elif dx <= dx_2:
                v_cmd = v * (dx - dx_1) / (dx_2 - dx_1)
            elif dx <= dx_3:
                v_cmd = v + (self.v_des - this_vel) * (dx - dx_2) \
                        / (dx_3 - dx_2)
            else:
                v_cmd = self.v_des

        edge = env.k.vehicle.get_edge(self.veh_id)

        if edge == "":
            return None

        if self.find_intersection_dist(env) <= 10 and \
                env.k.vehicle.get_edge(self.veh_id) in self.danger_edges or \
                env.k.vehicle.get_edge(self.veh_id)[0] == ":":
            return None
        else:
            # compute the acceleration from the desired velocity
            return (v_cmd - this_vel) / env.sim_step


class PISaturation(BaseController):
    """Inspired by Dan Work's... work.

    Dissipation of stop-and-go waves via control of autonomous vehicles:
    Field experiments https://arxiv.org/abs/1705.01693

    Usage
    -----
    See base class for example.

    Parameters
    ----------
    veh_id : str
        unique vehicle identifier
    car_following_params : flow.core.params.SumoCarFollowingParams
        object defining sumo-specific car-following parameters
    """

    def __init__(self, veh_id, car_following_params):
        """Instantiate PISaturation."""
        BaseController.__init__(self, veh_id, car_following_params, delay=1.0)

        # maximum achievable acceleration by the vehicle
        self.max_accel = car_following_params.controller_params['accel']

        # history used to determine AV desired velocity
        self.v_history = []

        # other parameters
        self.gamma = 2
        self.g_l = 7
        self.g_u = 30
        self.v_catch = 1

        # values that are updated by using their old information
        self.alpha = 0
        self.beta = 1 - 0.5 * self.alpha
        self.U = 0
        self.v_target = 0
        self.v_cmd = 0

    def get_accel(self, env):
        """See parent class."""
        lead_id = env.k.vehicle.get_leader(self.veh_id)
        lead_vel = env.k.vehicle.get_speed(lead_id)
        this_vel = env.k.vehicle.get_speed(self.veh_id)

        dx = env.k.vehicle.get_headway(self.veh_id)
        dv = lead_vel - this_vel
        dx_s = max(2 * dv, 4)

        # update the AV's velocity history
        self.v_history.append(this_vel)

        if len(self.v_history) == int(38 / env.sim_step):
            del self.v_history[0]

        # update desired velocity values
        v_des = np.mean(self.v_history)
        v_target = v_des + self.v_catch \
            * min(max((dx - self.g_l) / (self.g_u - self.g_l), 0), 1)

        # update the alpha and beta values
        alpha = min(max((dx - dx_s) / self.gamma, 0), 1)
        beta = 1 - 0.5 * alpha

        # compute desired velocity
        self.v_cmd = beta * (alpha * v_target + (1 - alpha) * lead_vel) \
            + (1 - beta) * self.v_cmd

        # compute the acceleration
        accel = (self.v_cmd - this_vel) / env.sim_step

        return min(accel, self.max_accel)
