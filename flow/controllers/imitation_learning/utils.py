import tensorflow as tf
import os
import numpy as np
import math
from flow.core.params import SumoCarFollowingParams
from flow.controllers.imitation_learning.imitating_controller import ImitatingController
from flow.controllers.imitation_learning.imitating_network import ImitatingNetwork
from flow.controllers.car_following_models import IDMController, SimCarFollowingController
from flow.core.rewards import *
from flow.envs.multiagent.bayesian_0_no_grid_env import *

""" Class agnostic helper functions """

def sample_trajectory_singleagent(env, controllers, action_network, max_trajectory_length, use_expert, max_decel):
    """
    Samples a single trajectory from a singleagent environment.
    Parameters
    __________
    env: gym.Env
        environment
    controllers: dict
        Dictionary of 2-tuples (Imitating_Controller, Expert_Controller), with keys of vehicle_ids
    action_network: ImitatingNetwork
        ImitatingNetwork class containing neural net for action prediction
    max_trajectory_length: int
        maximum steps in a trajectory
    use_expert: bool
        if True, trajectory is collected using expert policy (for behavioral cloning)
    max_decel: float
        maximum deceleration of environment. Used to determine dummy values to put as labels when environment has less vehicles than the maximum amount.
    Returns
    _______
    dict
        Dictionary of numpy arrays, where matching indeces of each array given (state, action, expert_action, reward, next_state, terminal) tuples
    """
    # reset and initialize arrays to store trajectory
    observation = env.reset()

    observations, actions, expert_actions, rewards, next_observations, terminals = [], [], [], [], [], []
    traj_length = 0

    while True:

        # update vehicle ids: if multidimensional action space, check if env has a sorted_rl_ids method
        if env.action_space.shape[0] > 1:
            try:
                vehicle_ids = env.get_sorted_rl_ids()
            except:
                vehicle_ids = env.k.vehicle.get_rl_ids()
        else:
            vehicle_ids = env.k.vehicle.get_rl_ids()

        # no RL actions if no RL vehicles
        if len(vehicle_ids) == 0:
            observation, reward, done, _ = env.step(None)
            if done:
                break
            continue

        # init controllers if any of vehicle ids are new
        # there could be multiple vehicle ids if they all share one state but have different actions
        car_following_params = SumoCarFollowingParams()

        for vehicle_id in vehicle_ids:
            if vehicle_id not in set(controllers.keys()):
                expert = IDMController(vehicle_id, car_following_params=car_following_params)
                imitator = ImitatingController(vehicle_id, action_network, False, car_following_params=car_following_params)
                controllers[vehicle_id] = (imitator, expert)


        # get the actions given by controllers
        action_dim = env.action_space.shape[0]
        rl_actions = []
        actions_expert = []

        invalid_expert_action = False
        for i in range(action_dim):
            # if max number of RL vehicles is not reached, insert dummy values
            if i >= len(vehicle_ids):
                # dummy value is -2 * max_decel
                ignore_accel = -2 * max_decel
                rl_actions.append(ignore_accel)
                actions_expert.append(ignore_accel)
            else:
                imitator = controllers[vehicle_ids[i]][0]
                expert = controllers[vehicle_ids[i]][1]
                expert_action = expert.get_action(env)
                # catch invalid expert actions
                if (expert_action is None or math.isnan(expert_action)):
                    invalid_expert_action = True

                actions_expert.append(expert_action)

                if use_expert:
                    if traj_length == 0 and i == 0:
                        print("Controller collecting trajectory: ", type(expert))
                    rl_actions.append(expert_action)
                else:
                    if traj_length == 0 and i == 0:
                        print("Controller collecting trajectory: ", type(imitator))
                    imitator_action = imitator.get_action(env)
                    rl_actions.append(imitator_action)


        # invalid action in rl_actions; default to Sumo, ignore sample
        if None in rl_actions or np.nan in rl_actions:
            observation, reward, done, _ = env.step(None)
            terminate_rollout = traj_length == max_trajectory_length or done
            if terminate_rollout:
                break
            continue
        # invalid expert action (if rl_actions is expert actions then this would have been caught above))
        if not use_expert and invalid_expert_action:
            # throw away sample, but step according to rl_actions
            observation, reward, done, _ = env.step(rl_actions)
            terminate_rollout = traj_length == max_trajectory_length or done
            if terminate_rollout:
                break
            continue

        # update collected data
        observations.append(observation)
        actions.append(rl_actions)
        expert_actions.append(actions_expert)
        observation, reward, done, _ = env.step(rl_actions)

        traj_length += 1
        next_observations.append(observation)
        rewards.append(reward)
        terminate_rollout = (traj_length == max_trajectory_length) or done
        terminals.append(terminate_rollout)

        if terminate_rollout:
            break

    return traj_dict(observations, actions, expert_actions, rewards, next_observations, terminals), traj_length


def sample_trajectory_multiagent(env, controllers, action_network, max_trajectory_length, use_expert, expert_control="SUMO"):
    """
    Samples a single trajectory from a multiagent environment.

    Parameters
    __________
    env: gym.Env
        environment
    controllers: dict
        Dictionary of 2-tuples (Imitating_Controller, Expert_Controller), with keys of vehicle_ids
    action_network: ImitatingNetwork
        ImitatingNetwork class containing neural net for action prediction
    max_trajectory_length: int
        maximum steps in a trajectory
    use_expert: bool
        if True, trajectory is collected using expert policy (for behavioral cloning)
    Returns
    _______
    dict
        Dictionary of numpy arrays, where matching indices of each array given (state, action, expert_action, reward, next_state, terminal, state_info) tuples
        state_info: dict
            state_info[veh_id] = [veh_edge, veh_route, veh_pos]

    """

    observation_dict = env.reset()

    observations, actions, expert_actions, rewards, next_observations, terminals, state_infos = [], [], [], [], [], [], []
    # account for get_acceleration being one step behind

    traj_length = 0
    done = {"__all__": False}

    while True:

        traj_length += 1
        terminate_rollout = done['__all__'] or (traj_length == max_trajectory_length)
        if terminate_rollout:
            break

        vehicle_ids = list(observation_dict.keys())
        # add nothing to replay buffer if no vehicles
        if len(vehicle_ids) == 0:
            observation_dict, reward, done, _ = env.step(None)
            if done['__all__']:
                break
            continue

        # actions taken by collecting controller
        rl_actions = dict()
        invalid_expert_action = False
        # actions taken by expert
        expert_action_dict= dict()
        # state info for relevant vehicle
        state_info_dict= dict()


        for i in range(len(vehicle_ids)):
            vehicle_id = vehicle_ids[i]
            if vehicle_id not in set(controllers.keys()):
                car_following_params = SumoCarFollowingParams()
                if expert_control == "SUMO":
                    expert = SimCarFollowingController(vehicle_id, car_following_params=car_following_params)
                else:
                    # Other controllers would just crash
                    expert = IDMController(vehicle_id, car_following_params=car_following_params)

                imitator = ImitatingController(vehicle_id, action_network, multiagent=True, car_following_params=car_following_params)
                controllers[vehicle_id] = (imitator, expert)

            expert_controller = controllers[vehicle_id][1]

            if use_expert:
                controller = expert_controller
            else:
                controller = controllers[vehicle_id][0]

            if actions == [] and i == 0:
                print("Controller collecting trajectory: ", controller)

            if expert_control == "SUMO" and use_expert:
                # SUMO gives the previous action taken
                action = env.k.vehicle.get_acceleration(vehicle_id)
                expert_action = action
                expert_action_dict[vehicle_id] = expert_action

            elif expert_control == "SUMO" and not use_expert:
                # for eg IDM, the acceleration 'action' is what the car takes on - no hard coding please TODO(KL)
                if vehicle_id == 'av_0':
                    if vehicle_id in env.inside_intersection and vehicle_id in env.past_intersection_rewarded_set:
                        action = None
                    elif env.arrived_intersection(vehicle_id) and not env.past_intersection(vehicle_id):
                        action = controller.get_action(env, allow_junction_control=True)
                    else:
                        action = None
                else:
                    action = None

                expert_action = env.k.vehicle.get_acceleration(vehicle_id)
                expert_action_dict[vehicle_id] = expert_action

            veh_edge, veh_route, veh_pos, edge_len = env.k.vehicle.get_edge(vehicle_id), env.k.vehicle.get_route(vehicle_id), \
                                                     env.k.vehicle.get_position(vehicle_id), env.k.network.edge_length(env.k.vehicle.get_edge(vehicle_id))
            state_info = [veh_edge, veh_route, veh_pos, edge_len]
            state_info_dict[vehicle_id] = state_info

            # action should be a scalar acceleration
            if type(action) == np.ndarray:
                action = action.flatten()[0]

            if (expert_action is None or math.isnan(expert_action)):
                invalid_expert_action = True
            
            rl_actions[vehicle_id] = action
            # print(f'rl_actions is {rl_actions}')

        if invalid_expert_action:
            # invalid action in rl_actions, so default control to SUMO
            observation_dict, reward_dict, done_dict, _ = env.step(None)
            terminate_rollout = traj_length == max_trajectory_length or done_dict['__all__']
            if terminate_rollout:
                break
            continue

        for vehicle_id in vehicle_ids:
            if not env.past_intersection(vehicle_id):
                observations.append(observation_dict[vehicle_id])
                actions.append(rl_actions[vehicle_id])
                expert_actions.append(expert_action_dict[vehicle_id])
                state_infos.append(state_info_dict[vehicle_id])

        if expert_control == "SUMO" and use_expert:
            observation_dict, reward_dict, done_dict, _ = env.step(None)
        if expert_control == "SUMO" and not use_expert:
            if not env.past_intersection("av_0"):
                try:
                    observation_dict, reward_dict, done_dict, _ = env.step(rl_actions)
                except:
                    import ipdb; ipdb.set_trace()
                    observation_dict, reward_dict, done_dict, _ = env.step(rl_actions)
            else:
                observation_dict, reward_dict, done_dict, _ = env.step(None)

        terminate_rollout = done_dict['__all__'] or (traj_length == max_trajectory_length)

        for vehicle_id in vehicle_ids:
            next_observations.append(observation_dict.get(vehicle_id, None))
            rewards.append(reward_dict.get(vehicle_id, 0))
            terminals.append(terminate_rollout)

        if terminate_rollout:
            break

    # deal with the fact that I'm retrieving the previous timestep from SUMO
    if use_expert and expert_control == "SUMO":
        expert_actions = expert_actions[1:]
        actions = expert_actions
        observations = observations[:-1]
        next_observations = next_observations[:-1]
        state_infos = state_infos[:-1]
        rewards = rewards[:-1]
    if not use_expert and expert_control == "SUMO":
        expert_actions = expert_actions[1:]
        actions = actions[:-1]
        observations = observations[:-1]
        next_observations = next_observations[:-1]
        state_infos = state_infos[:-1]
        rewards = rewards[:-1]

    return traj_dict(observations, actions, expert_actions, rewards, next_observations, terminals, state_infos), traj_length


def sample_trajectories(env, controllers, action_network, min_batch_timesteps, max_trajectory_length, multiagent, use_expert, max_decel=4.5):
    """
    Samples trajectories from environment.

    Parameters
    __________
    env: gym.Env
        environment
    controllers: dict
        Dictionary of 2-tuples (Imitating_Controller, Expert_Controller), with keys of vehicle_ids
    action_network: ImitatingNetwork
        ImitatingNetwork class containing neural net for action prediction
    min_batch_timesteps: int
        minimum number of env transitions to collect
    max_trajectory_length: int
        maximum steps in a trajectory
    multiagent: bool
        if True, env is a multiagent env
    use_expert: bool
        if True, trajectory is collected using expert policy (for behavioral cloning)
    max_decel: float
        maximum deceleration of environment. Used to determine dummy values to put as labels when environment has less vehicles than the maximum amount.

    Returns
    _______
    dict, int
        Dictionary of trajectory numpy arrays, where matching indeces of each array given (state, action, expert_action, reward, next_state, terminal, state_info) tuples
        Total number of env transitions seen over trajectories
    """
    total_envsteps = 0
    trajectories = []

    while total_envsteps < min_batch_timesteps:
        if multiagent:
            trajectory, traj_length = sample_trajectory_multiagent(env, controllers, action_network, max_trajectory_length, use_expert)
        else:
            trajectory, traj_length = sample_trajectory_singleagent(env, controllers, action_network, max_trajectory_length, use_expert, max_decel)

        trajectories.append(trajectory)

        total_envsteps += traj_length

    return trajectories, total_envsteps

def sample_n_trajectories(env, controllers, action_network, n, max_trajectory_length, multiagent, use_expert, max_decel=4.5):
    """
    Samples n trajectories from environment.

    Parameters
    __________
    env: gym.Env
        environment
    controllers: dict
        Dictionary of 2-tuples (Imitating_Controller, Expert_Controller), with keys of vehicle_ids
    action_network: ImitatingNetwork
        ImitatingNetwork class containing neural net for action prediction
    n: int
        number of trajectories to collect
    max_trajectory_length: int
        maximum steps in a trajectory
    multiagent: bool
        if True, env is a multiagent env
    use_expert: bool
        if True, trajectory is collected using expert policy (for behavioral cloning)
    max_decel: float
        maximum deceleration of environment. Used to determine dummy values to put as labels when environment has less vehicles than the maximum amount.

    Returns
    _______
    dict
        Dictionary of trajectory numpy arrays, where matching indeces of each array given (state, action, expert_action, reward, next_state, terminal) tuples
    """

    trajectories = []
    for _ in range(n):

        if multiagent:
            trajectory, length = sample_trajectory_multiagent(env, controllers, action_network, max_trajectory_length, use_expert)
        else:
            trajectory, length = sample_trajectory_singleagent(env, controllers, action_network, max_trajectory_length, use_expert, max_decel)

        trajectories.append((trajectory, length))

    return trajectories


def traj_dict(observations, actions, expert_actions, rewards, next_observations, terminals, state_infos):
    """
    Collects  observation, action, expert_action, rewards, next observation, terminal lists (collected over a rollout) into a single rollout dictionary.
    Parameters
    __________
    observations: list
        list of observations; ith entry is ith observation
    actions: list
        list of actions; ith entry is action taken at ith timestep
    rewards: list
        list of rewards; ith entry is reward received at ith timestep
    next_observations: list
        list of next observations; ith entry is the observation transitioned to due to state and action at ith timestep
    terminals: list
        list of booleans indicating if rollout ended at that timestep
    state_infos: list
        list of np arrays, each array is a list of [veh_edge, veh_route, veh_pos, edge_len]

    Returns
    _______
    dict
        dictionary containing above lists in numpy array form.
    """
    return {"observations" : np.array(observations),
            "actions" : np.array(actions),
            "expert_actions": np.array(expert_actions),
            "rewards" : np.array(rewards),
            "next_observations": np.array(next_observations),
            "terminals": np.array(terminals),
            "state_infos": np.array(state_infos)}
