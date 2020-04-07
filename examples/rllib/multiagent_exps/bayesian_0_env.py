import json
import argparse
import numpy as np

import ray
try:
    from ray.rllib.agents.agent import get_agent_class
except ImportError:
    from ray.rllib.agents.registry import get_agent_class
from ray.rllib.agents.ddpg.td3 import TD3Trainer
from ray.rllib.agents.ppo.ppo_policy import PPOTFPolicy
from ray import tune
from ray.tune.registry import register_env
from ray.tune import run_experiments

from flow.envs.multiagent import Bayesian1Env
from flow.networks import Bayesian0Network
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.params import SumoCarFollowingParams, VehicleParams
from flow.core.params import PedestrianParams

from flow.controllers import SimCarFollowingController, GridRouter, RLController


from flow.utils.registry import make_create_env
from flow.utils.rllib import FlowParamsEncoder

# Experiment parameters
N_ROLLOUTS = 20  # number of rollouts per training iteration
N_CPUS = 8 # number of parallel workers

# Environment parameters
HORIZON = 500  # time horizon of a single rollout
# TODO(@klin) make sure these parameters match what you've set up in the SUMO version here
V_ENTER = 30  # enter speed for departing vehicles
INNER_LENGTH = 50  # length of inner edges in the traffic light grid network
# number of vehicles originating in the left, right, top, and bottom edges
N_LEFT, N_RIGHT, N_TOP, N_BOTTOM = 0, 1, 1, 1


def make_flow_params(pedestrians=False, render=False):
    """
    Generate the flow params for the experiment.

    Parameters
    ----------

    Returns
    -------
    dict
        flow_params object
    """

    pedestrian_params = None
    if pedestrians:
        pedestrian_params = PedestrianParams()
        pedestrian_params.add(
            ped_id='ped_0',
            depart_time='0.00',
            start='(1.2)--(1.1)',
            end='(1.1)--(1.0)',
            depart_pos='43')

    # we place a sufficient number of vehicles to ensure they confirm with the
    # total number specified above. We also use a "right_of_way" speed mode to
    # support traffic light compliance
    vehicles = VehicleParams()
    # vehicles.add(
    #     veh_id="human",
    #     acceleration_controller=(SimCarFollowingController, {}),
    #     car_following_params=SumoCarFollowingParams(
    #         min_gap=2.5,
    #         max_speed=V_ENTER,
    #         decel=7.5,  # avoid collisions at emergency stops
    #         speed_mode="right_of_way",
    #     ),
    #     routing_controller=(GridRouter, {}),
    #     num_vehicles=1)

    #TODO(klin) make sure the autonomous vehicle being placed here is placed in the right position
    vehicles.add(
        veh_id='rl',
        acceleration_controller=(RLController, {}),
        car_following_params=SumoCarFollowingParams(
            speed_mode='right_of_way',
        ),
        routing_controller=(GridRouter, {}),
        num_vehicles=1)

    # vehicles.add(
    #     veh_id="human_1",
    #     acceleration_controller=(SimCarFollowingController, {}),
    #     car_following_params=SumoCarFollowingParams(
    #         min_gap=2.5,
    #         max_speed=V_ENTER,
    #         decel=7.5,  # avoid collisions at emergency stops
    #         speed_mode="right_of_way",
    #     ),
    #     routing_controller=(GridRouter, {}),
    #     num_vehicles=1)

    n_rows = 1
    n_columns = 1

    # define initial configs to pass into dict
    if pedestrians:
        initial_config = InitialConfig(
            spacing='custom',
            shuffle=False, 
            sidewalks=True, 
            lanes_distribution=float('inf'))
    else:
        initial_config = InitialConfig(
            spacing='custom',
            shuffle=False)

    flow_params = dict(
        # name of the experiment
        exp_tag="bayesian_0_env",

        # name of the flow environment the experiment is running on
        env_name=Bayesian1Env,

        # name of the network class the experiment is running on 
        network=Bayesian0Network,

        # simulator that is used by the experiment
        simulator='traci',

        # sumo-related parameters (see flow.core.params.SumoParams)
        sim=SumoParams(
            restart_instance=True,
            sim_step=0.1,
            render=render,
        ),

        # environment related parameters (see flow.core.params.EnvParams)
        env=EnvParams(
            horizon=HORIZON,
            additional_params={
                # maximum acceleration of autonomous vehicles
                'max_accel': 1,
                # maximum deceleration of autonomous vehicles
                'max_decel': 1,
                # desired velocity for all vehicles in the network, in m/s
                "target_velocity": 25,
                # how many objects in our local radius we want to return
                "max_num_objects": 3,
                # how large of a radius to search in for a given vehicle in meters
                "search_radius": 3000
            },
        ),

        # network-related parameters (see flow.core.params.NetParams and the
        # network's documentation or ADDITIONAL_NET_PARAMS component)
        net=NetParams(
            additional_params={
                "speed_limit": V_ENTER + 5,  # inherited from grid0 benchmark
                "grid_array": {
                    "inner_length": INNER_LENGTH,
                    "row_num": n_rows,
                    "col_num": n_columns,
                    "cars_left": N_LEFT,
                    "cars_right": N_RIGHT,
                    "cars_top": N_TOP,
                    "cars_bot": N_BOTTOM,
                },
                "horizontal_lanes": 1,
                "vertical_lanes": 1,
            },
        ),

        # vehicles to be placed in the network at the start of a rollout (see
        # flow.core.params.VehicleParams)
        veh=vehicles,

        ped=pedestrian_params,

        # parameters specifying the positioning of vehicles upon initialization
        # or reset (see flow.core.params.InitialConfig)
        initial = initial_config
    )

    return flow_params
def setup_exps_TD3(flow_params):
    """
    Experiment setup with TD3 using RLlib.

    Parameters
    ----------
    flow_params : dictionary of flow parameters

    Returns
    -------
    str
        name of the training algorithm
    str
        name of the gym environment to be trained
    dict
        training configuration parameters
    """
    alg_run = 'TD3'
    agent_cls = get_agent_class(alg_run)
    config = agent_cls._default_config.copy()
    config["num_workers"] = min(N_CPUS, N_ROLLOUTS)
    config['train_batch_size'] = HORIZON * N_ROLLOUTS
    # config['simple_optimizer'] = True
    config['gamma'] = 0.999  # discount rate
    config['model'].update({'fcnet_hiddens': [32, 32]})
    config['lr'] = tune.grid_search([1e-5, 1e-4, 1e-3])
    config['horizon'] = HORIZON
    config['observation_filter'] = 'NoFilter'

    # define callbacks for tensorboard

    def on_episode_start(info):
        env = info['env'].get_unwrapped()[0]
        episode = info['episode']
        episode.user_data['num_ped_collisions'] = 0
        episode.user_data['num_veh_collisions'] = 0
        episode.user_data['avg_speed'] = []

        episode.user_data['steps_elapsed'] = 0
        episode.user_data['vehicle_leaving_time'] = []
        episode.user_data['num_rl_veh_active'] = len(env.k.vehicle.get_rl_ids())

    def on_episode_step(info):
        env = info['env'].get_unwrapped()[0]
        episode = info['episode']
        for v_id in env.k.vehicle.get_rl_ids():
            if len(env.k.vehicle.get_pedestrian_crash(v_id, env.k.pedestrian)) > 0:
                episode.user_data['num_ped_collisions'] += 1

        if env.k.simulation.check_collision():
            episode.user_data['num_veh_collisions'] += 1

        avg_speed = env.k.vehicle.get_speed(env.k.vehicle.get_rl_ids())
        if len(avg_speed) > 0:
            avg_speed = np.mean(avg_speed)
            if avg_speed > 0:
                episode.user_data['avg_speed'].append(avg_speed)

        episode.user_data['steps_elapsed'] += 1
        num_veh_left = episode.user_data['num_rl_veh_active'] - len(env.k.vehicle.get_rl_ids())
        if num_veh_left > 0:
            episode.user_data['vehicle_leaving_time'] += \
                    [episode.user_data['steps_elapsed']] * num_veh_left
            episode.user_data['num_rl_veh_active'] -= num_veh_left

    def on_episode_end(info):
        env = info['env'].get_unwrapped()[0]
        episode = info['episode']
        episode.custom_metrics['num_ped_collisions'] = episode.user_data['num_ped_collisions']
        episode.custom_metrics['num_veh_collisions'] = episode.user_data['num_veh_collisions']
        episode.custom_metrics['avg_speed'] = np.mean(episode.user_data['avg_speed'])

        if episode.user_data['num_ped_collisions'] + episode.user_data['num_ped_collisions'] == 0:
            episode.user_data['vehicle_leaving_time'] += \
                    [episode.user_data['steps_elapsed']] * episode.user_data['num_rl_veh_active']
            episode.custom_metrics['avg_rl_veh_arrival'] = \
                np.mean(episode.user_data['vehicle_leaving_time'])
        else:
            episode.custom_metrics['avg_rl_veh_arrival'] = 500


    config['callbacks'] = {
            "on_episode_start":tune.function(on_episode_start),
            "on_episode_step":tune.function(on_episode_step),
            "on_episode_end":tune.function(on_episode_end)}

    # save the flow params for replay
    flow_json = json.dumps(
        flow_params, cls=FlowParamsEncoder, sort_keys=True, indent=4)
    config['env_config']['flow_params'] = flow_json
    config['env_config']['run'] = alg_run

    create_env, env_name = make_create_env(params=flow_params, version=0)

    # Register as rllib env
    register_env(env_name, create_env)

    test_env = create_env()
    obs_space = test_env.observation_space
    act_space = test_env.action_space

    def gen_policy():
        return None, obs_space, act_space, {}

    # Setup PG with a single policy graph for all agents
    policy_graphs = {'av': gen_policy()}

    def policy_mapping_fn(_):
        return 'av'

    config.update({
        'multiagent': {
            'policies': policy_graphs,
            'policy_mapping_fn': tune.function(policy_mapping_fn),
            'policies_to_train': ['av']
        }
    })

    return alg_run, env_name, config



def setup_exps_PPO(flow_params):
    """
    Experiment setup with PPO using RLlib.

    Parameters
    ----------
    flow_params : dictionary of flow parameters

    Returns
    -------
    str
        name of the training algorithm
    str
        name of the gym environment to be trained
    dict
        training configuration parameters
    """
    alg_run = 'PPO'
    agent_cls = get_agent_class(alg_run)
    config = agent_cls._default_config.copy()
    config["num_workers"] = min(N_CPUS, N_ROLLOUTS)
    config['train_batch_size'] = HORIZON * N_ROLLOUTS
    config['simple_optimizer'] = True
    config['gamma'] = 0.999  # discount rate
    config['model'].update({'fcnet_hiddens': [32, 32]})
    config['lr'] = tune.grid_search([1e-3, 1e-4, 1e-5])
    config['horizon'] = HORIZON
    config['observation_filter'] = 'NoFilter'

    # define callbacks for tensorboard

    def on_episode_start(info):
        env = info['env'].get_unwrapped()[0]
        episode = info['episode']
        episode.user_data['num_ped_collisions'] = 0
        episode.user_data['num_veh_collisions'] = 0
        episode.user_data['avg_speed'] = []

        episode.user_data['steps_elapsed'] = 0
        episode.user_data['vehicle_leaving_time'] = []
        episode.user_data['num_rl_veh_active'] = len(env.k.vehicle.get_rl_ids())

    def on_episode_step(info):
        env = info['env'].get_unwrapped()[0]
        episode = info['episode']
        for v_id in env.k.vehicle.get_rl_ids():
            if len(env.k.vehicle.get_pedestrian_crash(v_id, env.k.pedestrian)) > 0:
                episode.user_data['num_ped_collisions'] += 1

        if env.k.simulation.check_collision():
            episode.user_data['num_veh_collisions'] += 1

        avg_speed = env.k.vehicle.get_speed(env.k.vehicle.get_rl_ids())
        if len(avg_speed) > 0:
            avg_speed = np.mean(avg_speed)
            if avg_speed > 0:
                episode.user_data['avg_speed'].append(avg_speed)

        episode.user_data['steps_elapsed'] += 1
        num_veh_left = episode.user_data['num_rl_veh_active'] - len(env.k.vehicle.get_rl_ids())
        if num_veh_left > 0:
            episode.user_data['vehicle_leaving_time'] += \
                    [episode.user_data['steps_elapsed']] * num_veh_left
            episode.user_data['num_rl_veh_active'] -= num_veh_left

    def on_episode_end(info):
        env = info['env'].get_unwrapped()[0]
        episode = info['episode']
        episode.custom_metrics['num_ped_collisions'] = episode.user_data['num_ped_collisions']
        episode.custom_metrics['num_veh_collisions'] = episode.user_data['num_veh_collisions']
        episode.custom_metrics['avg_speed'] = np.mean(episode.user_data['avg_speed'])

        if episode.user_data['num_ped_collisions'] + episode.user_data['num_ped_collisions'] == 0:
            episode.user_data['vehicle_leaving_time'] += \
                    [episode.user_data['steps_elapsed']] * episode.user_data['num_rl_veh_active']
            episode.custom_metrics['avg_rl_veh_arrival'] = \
                np.mean(episode.user_data['vehicle_leaving_time'])
        else:
            episode.custom_metrics['avg_rl_veh_arrival'] = 500


    config['callbacks'] = {
            "on_episode_start":tune.function(on_episode_start),
            "on_episode_step":tune.function(on_episode_step),
            "on_episode_end":tune.function(on_episode_end)}

    # save the flow params for replay
    flow_json = json.dumps(
        flow_params, cls=FlowParamsEncoder, sort_keys=True, indent=4)
    config['env_config']['flow_params'] = flow_json
    config['env_config']['run'] = alg_run

    create_env, env_name = make_create_env(params=flow_params, version=0)

    # Register as rllib env
    register_env(env_name, create_env)

    test_env = create_env()
    obs_space = test_env.observation_space
    act_space = test_env.action_space

    def gen_policy():
        return PPOTFPolicy, obs_space, act_space, {}

    # Setup PG with a single policy graph for all agents
    policy_graphs = {'av': gen_policy()}

    def policy_mapping_fn(_):
        return 'av'

    config.update({
        'multiagent': {
            'policies': policy_graphs,
            'policy_mapping_fn': tune.function(policy_mapping_fn),
            'policies_to_train': ['av']
        }
    })

    return alg_run, env_name, config

def setup_exps_QMIX(args, flow_params):
    """
    Experiment setup with PPO using RLlib.

    Parameters
    ----------
    flow_params : dictionary of flow parameters

    Returns
    -------
    str
        name of the training algorithm
    str
        name of the gym environment to be trained
    dict
        training configuration parameters
    """
    alg_run = 'contrib/MADDPG'
    agent_cls = get_agent_class(alg_run)
    config = agent_cls._default_config.copy()
    config['no_done_at_end'] = True
    config['gamma'] = 0.999  # discount rate
    # if args.grid_search:
    #     config['lr'] = tune.grid_search([1e-3, 1e-4, 1e-5])
    #     config['buffer_size'] = tune.grid_search([10000, 100000])
    config['horizon'] = args.horizon
    config['observation_filter'] = 'NoFilter'

    # define callbacks for tensorboard

    config['callbacks'] = {
            "on_episode_start":tune.function(on_episode_start),
            "on_episode_step":tune.function(on_episode_step),
            "on_episode_end":tune.function(on_episode_end)}

    # save the flow params for replay
    flow_json = json.dumps(
        flow_params, cls=FlowParamsEncoder, sort_keys=True, indent=4)
    config['env_config']['flow_params'] = flow_json
    config['env_config']['run'] = alg_run

    create_env, env_name = make_create_env(params=flow_params, version=0)

    # Register as rllib env
    register_env(env_name, create_env)

    env = create_env()
    observation_space_dict = {i: env.observation_space for i in range(env.max_num_agents)}
    action_space_dict = {i: env.action_space for i in range(env.max_num_agents)}

    def gen_policy(i):
        return (
            None,
            env.observation_space,
            env.action_space,
            {
                "agent_id": i,
                "use_local_critic": True,
                "obs_space_dict": observation_space_dict,
                "act_space_dict": action_space_dict,
            }
        )

    policies = {"policy_%d" %i: gen_policy(i) for i in range(env.max_num_agents)}
    policy_ids = list(policies.keys())
    config.update({"multiagent": {
                    "policies": policies,
                    "policy_mapping_fn": ray.tune.function(
                        lambda i: policy_ids[i]
                    )
                }
    })

    return alg_run, env_name, config

if __name__ == '__main__':
    EXAMPLE_USAGE = """
    example usage:
        python multiagent_traffic_light_grid.py --upload_dir=<S3 bucket>
    """

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="[Flow] Issues multi-agent traffic light grid experiment",
        epilog=EXAMPLE_USAGE)

    # required input parameters
    parser.add_argument("--upload_dir", type=str,
                        help="S3 Bucket for uploading results.")

    # optional input parameters
    parser.add_argument('--grid_search', action='store_true', default=False,
                        help='If true, a grid search is run')
    parser.add_argument('--run_mode', type=str, default='local',
                        help="Experiment run mode (local | cluster)")
    parser.add_argument('--algo', type=str, default='QMIX',
                        help="RL method to use (PPO, TD3, QMIX)")
    parser.add_argument("--pedestrians",
                        help="use pedestrians, sidewalks, and crossings in the simulation",
                        action="store_true")
    args = parser.parse_args()

    pedestrians = args.pedestrians
    render = args.render

    flow_params = make_flow_params(pedestrians, render)

    upload_dir = args.upload_dir
    RUN_MODE = args.run_mode
    ALGO = args.algo

    if ALGO == 'PPO':
        alg_run, env_name, config = setup_exps_PPO(args, flow_params)
    elif ALGO == 'TD3':
        alg_run, env_name, config = setup_exps_TD3(args, flow_params)
    elif ALGO == 'QMIX':
        alg_run, env_name, config = setup_exps_QMIX(args, flow_params)
    else:
        raise NotImplementedError

    if RUN_MODE == 'local':
        ray.init(num_cpus=N_CPUS + 1)
        N_ITER = 1
    elif RUN_MODE == 'cluster':
        ray.init(redis_address="localhost:6379")
        N_ITER = 2000

    exp_tag = {
        'run': alg_run,
        'env': env_name,
        'checkpoint_freq': 25,
        "max_failures": 10,
        'stop': {
            'training_iteration': 100
        },
        'config': config,
        "num_samples": 1,
    }

    if upload_dir:
        exp_tag["upload_dir"] = "s3://{}".format(upload_dir)

    run_experiments(
        {
            flow_params["exp_tag"]: exp_tag
         },
    )
