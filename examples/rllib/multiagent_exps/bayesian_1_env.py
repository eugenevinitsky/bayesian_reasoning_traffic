import argparse
import json
import numpy as np

from gym.spaces import Tuple
import ray
try:
    from ray.rllib.agents.agent import get_agent_class
except ImportError:
    from ray.rllib.agents.registry import get_agent_class
from ray.rllib.agents.ddpg.td3 import TD3Trainer
from ray.rllib.env.group_agents_wrapper import _GroupAgentsWrapper
from ray.rllib.agents.ppo.ppo_policy import PPOTFPolicy
from ray import tune
from ray.tune.registry import register_env
from ray.tune import run_experiments

from flow.envs.multiagent import Bayesian1Env
from flow.networks import Bayesian1Network
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.params import SumoCarFollowingParams, VehicleParams
from flow.core.params import PedestrianParams

from flow.controllers import SimCarFollowingController, GridRouter, RLController


from flow.utils.registry import make_create_env
from flow.utils.rllib import FlowParamsEncoder


# Environment parameters
# TODO(@klin) make sure these parameters match what you've set up in the SUMO version here
V_ENTER = 30  # enter speed for departing vehicles
INNER_LENGTH = 50  # length of inner edges in the traffic light grid network
# number of vehicles originating in the left, right, top, and bottom edges
N_LEFT, N_RIGHT, N_TOP, N_BOTTOM = 0, 1, 1, 1


def make_flow_params(args, pedestrians=False):
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
        for i in range(1):
            name = "ped_" + str(i)
            time = str(i * 5) + '.00'
            pedestrian_params.add(
                ped_id=name,
                depart_time=time,
                start='(1.1)--(2.1)',
                end='(2.1)--(1.1)',
                depart_pos='5',
                arrival_pos='43')

    # we place a sufficient number of vehicles to ensure they confirm with the
    # total number specified above. We also use a "right_of_way" speed mode to
    # support traffic light compliance
    vehicles = VehicleParams()

    #TODO(klin) make sure the autonomous vehicle being placed here is placed in the right position

    vehicles.add(
        veh_id="human",
        acceleration_controller=(SimCarFollowingController, {}),
        car_following_params=SumoCarFollowingParams(
            min_gap=2.5,
            max_speed=V_ENTER,
            decel=7.5,  # avoid collisions at emergency stops
            speed_mode="right_of_way",
        ),
        routing_controller=(GridRouter, {}),
        num_vehicles=0)

    vehicles.add(
        veh_id='rl',
        acceleration_controller=(RLController, {}),
        car_following_params=SumoCarFollowingParams(
            speed_mode="right_of_way",
        ),
        routing_controller=(GridRouter, {}),
        num_vehicles=3)

    '''
    vehicles.add(
        veh_id="human_1",
        acceleration_controller=(SimCarFollowingController, {}),
        car_following_params=SumoCarFollowingParams(
            min_gap=2.5,
            max_speed=V_ENTER,
            decel=7.5,  # avoid collisions at emergency stops
            speed_mode="right_of_way",
        ),
        routing_controller=(GridRouter, {}),
        num_vehicles=1)
    '''

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
        exp_tag="bayesian_1_env",

        # name of the flow environment the experiment is running on
        env_name=Bayesian1Env,

        # name of the network class the experiment is running on 
        network=Bayesian1Network,

        # simulator that is used by the experiment
        simulator='traci',

        # sumo-related parameters (see flow.core.params.SumoParams)
        sim=SumoParams(
            restart_instance=True,
            sim_step=0.1,
            render=False,
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
                "search_radius": 50,
                # whether we use the multi-agent algorithm QMIX
                "maddpg": args.algo == "MADDPG"
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

    episode.user_data['steps_elapsed'] = 0
    episode.user_data['vehicle_leaving_time'] = []
    episode.user_data['num_rl_veh_active'] = len(env.k.vehicle.get_rl_ids())

def on_episode_step(info):
    env = info['env'].get_unwrapped()[0]
    if isinstance(env, _GroupAgentsWrapper):
        env = env.env
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


def setup_exps_TD3(args, flow_params):
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
    # config['simple_optimizer'] = True
    config['gamma'] = 0.999  # discount rate
    config['model'].update({'fcnet_hiddens': [256, 256]})
    if args.grid_search:
        config['actor_lr'] = tune.grid_search([1e-5, 1e-4])
        config['critic_lr'] = tune.grid_search([1e-5, 1e-4])
        config['prioritized_replay'] = tune.grid_search([True, False])
    config['horizon'] = args.horizon
    config['no_done_at_end'] = True
    config['observation_filter'] = 'NoFilter'

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
    alg_run = 'PPO'
    agent_cls = get_agent_class(alg_run)
    config = agent_cls._default_config.copy()
    config["num_workers"] = min(args.n_cpus, args.n_rollouts)
    config['train_batch_size'] = args.horizon * args.n_rollouts
    config['simple_optimizer'] = True
    config['no_done_at_end'] = True
    config['gamma'] = 0.999  # discount rate
    config['model'].update({'fcnet_hiddens': [256, 256]})
    if args.grid_search:
        config['lr'] = tune.grid_search([1e-3, 1e-4, 1e-5])
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


def setup_exps_MADDPG(args, flow_params):
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

    from flow.algorithms.maddpg.maddpg import DEFAULT_CONFIG as MADDPG_DEFAULT_CONFIG, MADDPGTrainer
    alg_run = MADDPGTrainer
    config = MADDPG_DEFAULT_CONFIG.copy()
    config['no_done_at_end'] = True
    config['gamma'] = 0.95  # discount rate
    if args.grid_search:
        config['actor_lr'] = tune.grid_search([1e-2, 1e-3])
        config['critic_lr'] = tune.grid_search([1e-2, 1e-3])
        config['n_step'] = tune.grid_search([1, 10])
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
                "use_local_critic": False,
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
    parser.add_argument("--n_iterations", type=int, default=250,
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
                        help="RL method to use (PPO, TD3)")
    parser.add_argument("--pedestrians",
                        help="use pedestrians, sidewalks, and crossings in the simulation",
                        action="store_true")
    args = parser.parse_args()

    pedestrians = args.pedestrians
    flow_params = make_flow_params(args, pedestrians)

    upload_dir = args.upload_dir
    RUN_MODE = args.run_mode
    ALGO = args.algo
    CHECKPOINT_FREQ = args.checkpoint_freq

    if ALGO == 'PPO':
        alg_run, env_name, config = setup_exps_PPO(args, flow_params)
    elif ALGO == 'TD3':
        alg_run, env_name, config = setup_exps_TD3(args, flow_params)
    elif ALGO == 'MADDPG':
        alg_run, env_name, config = setup_exps_MADDPG(args, flow_params)
        CHECKPOINT_FREQ *= 10
    else:
        raise NotImplementedError

    if RUN_MODE == 'local' and not args.grid_search:
        ray.init(num_cpus=args.n_cpus + 1, local_mode=True)
    elif RUN_MODE == 'cluster':
        ray.init(redis_address="localhost:6379")

    exp_tag = {
        'run': alg_run,
        'env': env_name,
        'checkpoint_freq': CHECKPOINT_FREQ,
        "max_failures": 1,
        'stop': {
            'training_iteration': args.n_iterations
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
