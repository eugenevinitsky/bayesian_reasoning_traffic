import argparse
from datetime import datetime
import json
import numpy as np
import subprocess
import os

import pytz
import ray
try:
    from ray.rllib.agents.agent import get_agent_class
except ImportError:
    from ray.rllib.agents.registry import get_agent_class
from ray.rllib.env.group_agents_wrapper import _GroupAgentsWrapper
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy
from ray import tune
from ray.tune import run as run_tune
from ray.tune.registry import register_env

from flow.envs.multiagent import Bayesian0NoGridEnv
from flow.networks import Bayesian0Network
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.params import SumoCarFollowingParams, VehicleParams
from flow.core.params import PedestrianParams

from flow.controllers import SimCarFollowingController, GridRouter, RLController, IDMController
# from flow.controllers.car_following_models import IDMController


from flow.utils.registry import make_create_env
from flow.utils.rllib import FlowParamsEncoder

# Environment parameters
# TODO(@klin) make sure these parameters match what you've set up in the SUMO version here
V_ENTER = 0  # enter speed for departing vehicles
MAX_SPEED = 30
INNER_LENGTH = 50  # length of inner edges in the traffic light grid network
# number of vehicles originating in the left, right, top, and bottom edges
N_LEFT, N_RIGHT, N_TOP, N_BOTTOM = 0, 1, 1, 1
NUM_PEDS = 6


def make_flow_params(args, pedestrians=False, render=False, discrete=False):
    """
    Generate the flow params for the experiment.

    Parameters
    ----------

    Returns
    -------
    dict
        flow_params object
    """
    pedestrian_params = PedestrianParams()
    for i in range(NUM_PEDS):
        pedestrian_params.add(
            ped_id=f'ped_{i}',
            depart_time='0.00',
            start='(1.2)--(1.1)',
            end='(1.1)--(1.0)',
            depart_pos=f'{44 + 0.5*i}',
            arrival_pos='5')
        
    # we place a sufficient number of vehicles to ensure they confirm with the
    # total number specified above. We also use a "right_of_way" speed mode to
    # support traffic light compliance
    vehicles = VehicleParams()

    if args.only_rl:
        vehicles.add(
            veh_id='av',
            acceleration_controller=(RLController, {}),
            car_following_params=SumoCarFollowingParams(
                speed_mode='aggressive',
            ),
            routing_controller=(GridRouter, {}),
            # depart_time='3.5',    #TODO change back to 3.5s
            num_vehicles=4,
        )
    else:
        vehicles.add(
            veh_id="human_0",
            acceleration_controller=(SimCarFollowingController, {}),
            car_following_params=SumoCarFollowingParams(
                min_gap=2.5,
                decel=7.5,  # avoid collisions at emergency stops
                speed_mode="right_of_way",
            ),
            routing_controller=(GridRouter, {}),
            depart_time='0.25',
            num_vehicles=1)

        #TODO(klin) make sure the autonomous vehicle being placed here is placed in the right position
        vehicles.add(
            veh_id='av',
            acceleration_controller=(RLController, {}),
            car_following_params=SumoCarFollowingParams(
                speed_mode='aggressive',
            ),
            routing_controller=(GridRouter, {}),
            # depart_time='3.5',    #TODO change back to 3.5s
            num_vehicles=1,
            )
        if args.randomize_vehicles:
            vehicles.add(
                veh_id="human_1",
                acceleration_controller=(SimCarFollowingController, {}),
                car_following_params=SumoCarFollowingParams(
                    min_gap=2.5,
                    decel=7.5,  # avoid collisions at emergency stops
                    speed_mode="right_of_way",
                ),
                routing_controller=(GridRouter, {}),
                num_vehicles=1)

            vehicles.add(
                veh_id="human_2",
                acceleration_controller=(SimCarFollowingController, {}),
                car_following_params=SumoCarFollowingParams(
                    min_gap=2.5,
                    decel=7.5,  # avoid collisions at emergency stops
                    speed_mode="right_of_way",
                ),
                routing_controller=(GridRouter, {}),
                num_vehicles=1)


    n_rows = 1
    n_columns = 1

    # define initial configs to pass into dict
    if pedestrians:
        initial_config = InitialConfig(
            spacing='custom',
            shuffle=True,
            sidewalks=True, 
            lanes_distribution=float('inf'))
    else:
        initial_config = InitialConfig(
            spacing='custom',
            shuffle=False)

    flow_params = dict(
        # name of the experiment
        exp_tag="bayesian_0_no_grid_env",

        # name of the flow environment the experiment is running on
        env_name=Bayesian0NoGridEnv,

        # name of the network class the experiment is running on 
        network=Bayesian0Network,

        # simulator that is used by the experiment
        simulator='traci',

        # sumo-related parameters (see flow.core.params.SumoParams)
        sim=SumoParams(
            restart_instance=True,
            sim_step=0.2,
            render=render,
        ),

        # environment related parameters (see flow.core.params.EnvParams)
        
        env=EnvParams(
            horizon=args.horizon,
            additional_params={
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
                "discrete": discrete,
                # whether to randomize which edge the vehicles are coming from
                "randomize_vehicles": args.randomize_vehicles,
                # whether to append the prior into the state
                "inference_in_state": False,
                # whether to grid the cone "search_veh_radius" in front of us into 6 grid cells
                "use_grid": False
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
                "randomize_routes": True,
                # "vehicle_kernel": vehicles,
                # "pedestrian_kernel": pedestrian_params,
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

    # define callbacks for tensorboard


def on_episode_start(info):
    env = info['env'].get_unwrapped()[0]
    if isinstance(env, _GroupAgentsWrapper):
        env = env.env
    episode = info['episode']
    episode.user_data['num_ped_collisions'] = 0
    episode.user_data['num_veh_collisions'] = 0
    episode.user_data['avg_speed'] = []
    episode.user_data["discounted_reward"] = 0

    episode.user_data['steps_elapsed'] = 0
    episode.user_data['vehicle_leaving_time'] = []
    episode.user_data['num_rl_veh_active'] = len(env.k.vehicle.get_rl_ids())
    episode.user_data['past_intersection'] = 0

def on_episode_step(info):
    env = info['env'].get_unwrapped()[0]
    if isinstance(env, _GroupAgentsWrapper):
        env = env.env
    episode = info['episode']
    collide_ids = env.k.simulation.get_collision_vehicle_ids()
    for v_id in env.k.vehicle.get_rl_ids():
        if len(env.k.vehicle.get_pedestrian_crash(v_id, env.k.pedestrian)) > 0:
            episode.user_data['num_ped_collisions'] += 1

        if v_id in collide_ids:
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
    rl_ids = env.k.vehicle.get_rl_ids()
    if 'av_0' in rl_ids:
        episode.user_data['past_intersection'] = int(env.k.vehicle.get_route('av_0')[-1] == env.k.vehicle.get_edge('av_0'))
        # TODO(@evinitsky) remove hardcoding
    if 'av_0' in env.reward.keys():
        episode.user_data["discounted_reward"] += env.reward['av_0'] * (0.995 ** env.time_counter)


def on_episode_end(info):
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
    episode.custom_metrics['past_intersection'] = episode.user_data['past_intersection']
    episode.custom_metrics["discounted_reward"] = episode.user_data['discounted_reward']


def setup_exps_DQN(args, flow_params):
    """
    Experiment setup with DQN using RLlib.

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

    alg_run = 'DQN'
    agent_cls = get_agent_class(alg_run)
    config = agent_cls._default_config.copy()
    # print(config)
    # import ipdb; ipdb.set_trace()
    config["num_workers"] = min(args.n_cpus, args.n_rollouts)
    if args.render:
        config["num_workers"] = 0
    config['train_batch_size'] = args.horizon * args.n_rollouts
    config['no_done_at_end'] = False
    config['lr'] = 1e-4
    config['n_step'] = 10
    config['gamma'] = 0.995  # discount rate
    config['model'].update({'fcnet_hiddens': [64, 64, 64]})
    config['learning_starts'] = 20000
    config['prioritized_replay'] = True
    # increase buffer size
    config['buffer_size'] = 200000
    config["train_batch_size"] = 320
    # config['model']['fcnet_activation'] = 'relu'
    if args.grid_search:
        config['n_step'] = tune.grid_search([1, 10])
        # config['lr'] = tune.grid_search([1e-3, 1e-2, 1e-4])
        config["train_batch_size"] = tune.grid_search([32, 320])
        config['gamma'] = tune.grid_search([0.999, 0.99])  # discount rate

    config['horizon'] = args.horizon
    config['observation_filter'] = 'NoFilter'

    # define callbacks for tensorboard

    def on_train_result(info):
        result = info['result']
        trainer = info['trainer']
        trainer.workers.foreach_worker(
                lambda ev: ev.foreach_env(
                    lambda env: env.update_curriculum(result['training_iteration'])
                )
        )

    config['callbacks'] = {
            "on_episode_start":tune.function(on_episode_start),
            "on_episode_step":tune.function(on_episode_step),
            "on_episode_end":tune.function(on_episode_end),
            "on_train_result":tune.function(on_train_result)}

    # save the flow params for replay
    flow_json = json.dumps(
        flow_params, cls=FlowParamsEncoder, sort_keys=True, indent=4)
    config['env_config']['flow_params'] = flow_json
    config['env_config']['run'] = alg_run

    create_env, env_name = make_create_env(params=flow_params, version=0)
    config['env'] = env_name

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

def setup_exps_PPO(args, flow_params):
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

    # from flow.algorithms.ppo.ppo import DEFAULT_CONFIG as PPO_DEFAULT_CONFIG, PPOTrainer
    alg_run = "PPO"
    # config = PPO_DEFAULT_CONFIG.copy()
    agent_cls = get_agent_class(alg_run)
    config = agent_cls._default_config.copy()

    config["num_workers"] = min(args.n_cpus, args.n_rollouts)
    config['train_batch_size'] = args.horizon * args.n_rollouts
    config['simple_optimizer'] = False
    # TODO(@ev) fix the termination condition so you don't need this
    config['no_done_at_end'] = False
    config['lr'] = 1e-4
    config['gamma'] = 0.97  # discount rate
    # config['entropy_coeff'] = -0.01
    config['model'].update({'fcnet_hiddens': [256, 256]})
    if args.use_lstm:
        config['model']['use_lstm'] = True
    if args.grid_search:
        config['gamma'] = tune.grid_search([.995, 0.99, 0.9])  # discount rate
        # config['entropy_coeff'] = tune.grid_search([-0.005, -0.01, 0])  # entropy coeff

    config['horizon'] = args.horizon
    config['observation_filter'] = 'NoFilter'

    # define callbacks for tensorboard

    def on_train_result(info):
        result = info['result']
        trainer = info['trainer']
        trainer.workers.foreach_worker(
                lambda ev: ev.foreach_env(
                    lambda env: env.update_curriculum(result['training_iteration'])
                )
        )

    config['callbacks'] = {
            "on_episode_start":tune.function(on_episode_start),
            "on_episode_step":tune.function(on_episode_step),
            "on_episode_end":tune.function(on_episode_end),
            "on_train_result":tune.function(on_train_result)}

    # save the flow params for replay
    flow_json = json.dumps(
        flow_params, cls=FlowParamsEncoder, sort_keys=True, indent=4)
    config['env_config']['flow_params'] = flow_json
    config['env_config']['run'] = "PPO"

    create_env, env_name = make_create_env(params=flow_params, version=0)
    config['env'] = env_name

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
    parser.add_argument("--exp_title", type=str, default="test",
                        help="Where we store the experiment results")
    parser.add_argument("--upload_dir", type=str, default="bayesian-traffic",
                        help="S3 Bucket for uploading results.")
    parser.add_argument("--use_s3", action='store_true', default=False,
                        help="If true, upload to s3")
    parser.add_argument("--n_iterations", type=int, default=100,
                        help="Number of training iterations")
    parser.add_argument("--n_rollouts", type=int, default=20,
                        help="Number of rollouts per iteration")
    parser.add_argument("--checkpoint_freq", type=int, default=25,
                        help="How frequently to checkpoint")
    parser.add_argument("--n_cpus", type=int, default=1,
                        help="Number of rollouts per iteration")
    parser.add_argument("--horizon", type=int, default=500,
                        help="Horizon length of a rollout")

    # optional input parameters
    parser.add_argument('--grid_search', action='store_true', default=False,
                        help='If true, a grid search is run')
    parser.add_argument('--run_mode', type=str, default='local',
                        help="Experiment run mode (local | cluster)")
    parser.add_argument('--algo', type=str, default='PPO',
                        help="RL method to use (PPO, TD3, QMIX, DQN)")
    parser.add_argument("--pedestrians", default=True,
                        help="use pedestrians, sidewalks, and crossings in the simulation",
                        action="store_true")
    parser.add_argument("--randomize_vehicles", default=True,
                        help="randomize the number of vehicles in the system and where they come from",
                        action="store_true")
    parser.add_argument("--only_rl", default=False,
                        help="only use AVs in the system",
                        action="store_true")
    parser.add_argument("--render",
                        help="render SUMO simulation",
                        action="store_true")
    parser.add_argument("--discrete",
                        help="determine if policy has discrete actions",
                        action="store_true")
    parser.add_argument("--run_transfer_tests",
                        help="run the tests of generalization at the end of training",
                        action="store_true",
                        default=False)

    # Model arguments
    parser.add_argument("--use_lstm", action="store_true", default=False, help="Use LSTM")
    args = parser.parse_args()

    pedestrians = args.pedestrians
    render = args.render
    discrete = args.discrete
    if args.algo == 'DQN':
        discrete = True
    flow_params = make_flow_params(args, pedestrians, render, discrete)

    upload_dir = args.upload_dir
    RUN_MODE = args.run_mode
    ALGO = args.algo
    CHECKPOINT_FREQ = args.checkpoint_freq

    if ALGO == 'PPO':
        alg_run, env_name, config = setup_exps_PPO(args, flow_params)
    elif ALGO == 'DQN':
        alg_run, env_name, config = setup_exps_DQN(args, flow_params)

    else:
        raise NotImplementedError

    if RUN_MODE == 'local' and not args.grid_search:
        ray.init(num_cpus=args.n_cpus + 1, local_mode=True)
    elif RUN_MODE == 'cluster':
        ray.init(redis_address="localhost:6379")

    # create a custom string that makes looking at the experiment names easier
    def trial_str_creator(trial):
        return "{}_{}".format(trial.trainable_name, trial.experiment_tag)

    exp_dict = {
        'name': args.exp_title,
        'run_or_experiment': alg_run,
        # 'env': env_name,
        'checkpoint_freq': CHECKPOINT_FREQ,
        'trial_name_creator': trial_str_creator,
        "max_failures": 1,
        'stop': {
            'training_iteration': args.n_iterations
        },
        'config': config,
        "num_samples": 1,
    }

    if args.use_s3:
        date = datetime.now(tz=pytz.utc)
        date = date.astimezone(pytz.timezone('US/Pacific')).strftime("%m-%d-%Y")
        s3_string = os.path.join(os.path.join(upload_dir, date), args.exp_title)
        exp_dict["upload_dir"] = "s3://{}".format(s3_string)

    run_tune(**exp_dict, queue_trials=False, raise_on_failed_trial=False)

    if args.run_transfer_tests:
        from flow.visualize.transfer_test import create_parser, run_transfer
        for (dirpath, dirnames, filenames) in os.walk(os.path.expanduser("~/ray_results")):
            if "checkpoint_{}".format(args.n_iterations) in dirpath and dirpath.split('/')[-3] == args.exp_title:
                ray.shutdown()
                ray.init()
                parser = create_parser()
                folder = os.path.dirname(dirpath)
                tune_name = folder.split("/")[-1]
                temp_args = parser.parse_args([folder, str(args.n_iterations), "--num_rollouts", "100",
                                               "--render_mode", "no_render"])
                run_transfer(temp_args)

                if args.use_s3:
                    # visualize_adversaries(config, checkpoint_path, 10, 100, output_path)
                    for i in range(4):
                        try:
                            p1 = subprocess.Popen("aws s3 sync {} {}".format(os.path.expanduser("~/generalization"),
                                                                             "s3://bayesian-traffic/transfer_results/{}/{}/{}".format(
                                                                                 date,
                                                                                 args.exp_title,
                                                                                 tune_name)).split(
                                ' '))
                            p1.wait(50)
                        except Exception as e:
                            print('This is the error ', e)
