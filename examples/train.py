"""Runner script for single and multi-agent reinforcement learning experiments.

This script performs an RL experiment using the PPO algorithm. Choice of
hyperparameters can be seen and adjusted from the code below.

Usage
    python train.py EXP_CONFIG
"""
import argparse
from datetime import datetime
import json
import os
import sys
from time import strftime
from copy import deepcopy
import numpy as np
import pytz

from flow.core.util import ensure_dir
from flow.utils.registry import env_constructor
from flow.utils.rllib import FlowParamsEncoder, get_flow_params
from flow.utils.registry import make_create_env


def parse_args(args):
    """Parse training options user can specify in command line.

    Returns
    -------
    argparse.Namespace
        the output parser object
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Parse argument used when running a Flow simulation.",
        epilog="python train.py EXP_CONFIG")

    # required input parameters
    parser.add_argument(
        'exp_config', type=str,
        help='Name of the experiment configuration file, as located in '
             'exp_configs/rl/singleagent or exp_configs/rl/multiagent.')

    parser.add_argument(
        'exp_title', type=str,
        help='Name of experiment that results will be stored in')

    # optional input parameters
    parser.add_argument(
        '--rl_trainer', type=str, default="rllib",
        help='the RL trainer to use. either rllib or Stable-Baselines')
    parser.add_argument(
        '--load_weights_path', type=str, default=None,
        help='Path to h5 file containing a pretrained model. Relevent for PPO with RLLib'
    )
    parser.add_argument(
        '--algorithm', type=str, default="PPO",
        help='RL algorithm to use. Options are PPO, TD3, and CENTRALIZEDPPO (which uses a centralized value function)'
             ' right now.'
    )
    parser.add_argument(
        '--num_cpus', type=int, default=1,
        help='How many CPUs to use')
    parser.add_argument(
        '--num_steps', type=int, default=5000,
        help='How many total steps to perform learning over. Relevant for stable-baselines')
    parser.add_argument(
        '--grid_search', action='store_true', default=False,
        help='Whether to grid search over hyperparams')
    parser.add_argument(
        '--num_iterations', type=int, default=200,
        help='How many iterations are in a training run.')
    parser.add_argument(
        '--checkpoint_freq', type=int, default=20,
        help='How often to checkpoint.')
    parser.add_argument(
        '--num_rollouts', type=int, default=1,
        help='How many rollouts are in a training batch')
    parser.add_argument(
        '--rollout_size', type=int, default=1000,
        help='How many steps are in a training batch.')
    parser.add_argument('--use_s3', action='store_true', help='If true, upload results to s3')
    parser.add_argument('--local_mode', action='store_true', default=False,
                        help='If true only 1 CPU will be used')
    parser.add_argument('--render', action='store_true', default=False,
                        help='If true, we render the display')
    parser.add_argument(
        '--checkpoint_path', type=str, default=None,
        help='Directory with checkpoint to restore training from.')

    return parser.parse_known_args(args)[0]


def run_model_stablebaseline(flow_params,
                             num_cpus=1,
                             rollout_size=50,
                             num_steps=50):
    """Run the model for num_steps if provided.

    Parameters
    ----------
    flow_params : dict
        flow-specific parameters
    num_cpus : int
        number of CPUs used during training
    rollout_size : int
        length of a single rollout
    num_steps : int
        total number of training steps
    The total rollout length is rollout_size.

    Returns
    -------
    stable_baselines.*
        the trained model
    """
    if num_cpus == 1:
        constructor = env_constructor(params=flow_params, version=0)()
        # The algorithms require a vectorized environment to run
        env = DummyVecEnv([lambda: constructor])
    else:
        env = SubprocVecEnv([env_constructor(params=flow_params, version=i)
                             for i in range(num_cpus)])

    train_model = PPO2('MlpPolicy', env, verbose=1, n_steps=rollout_size)
    train_model.learn(total_timesteps=num_steps)
    return train_model


def setup_exps_rllib(flow_params,
                     n_cpus,
                     n_rollouts,
                     flags,
                     policy_graphs=None,
                     policy_mapping_fn=None,
                     policies_to_train=None,
                     ):
    """Return the relevant components of an RLlib experiment.

    Parameters
    ----------
    flow_params : dict
        flow-specific parameters (see flow/utils/registry.py)
    n_cpus : int
        number of CPUs to run the experiment over
    n_rollouts : int
        number of rollouts per training iteration
    flags:
        custom arguments
    policy_graphs : dict, optional
        TODO
    policy_mapping_fn : function, optional
        TODO
    policies_to_train : list of str, optional
        TODO
    Returns
    -------
    str
        name of the training algorithm
    str
        name of the gym environment to be trained
    dict
        training configuration parameters
    """
    from ray import tune
    from ray.tune.registry import register_env
    from ray.rllib.env.group_agents_wrapper import _GroupAgentsWrapper
    try:
        from ray.rllib.agents.agent import get_agent_class
    except ImportError:
        from ray.rllib.agents.registry import get_agent_class

    horizon = flow_params['env'].horizon

    alg_run = flags.algorithm.upper()

    if alg_run == "PPO":
        from flow.controllers.imitation_learning.custom_ppo import CustomPPOTrainer
        from ray.rllib.agents.ppo import DEFAULT_CONFIG
        config = deepcopy(DEFAULT_CONFIG)


        alg_run = CustomPPOTrainer

        horizon = flow_params['env'].horizon

        config["num_workers"] = n_cpus
        config["no_done_at_end"] = True
        config["horizon"] = horizon
        config["model"].update({"fcnet_hiddens": [32, 32, 32]})
        config["train_batch_size"] = horizon * n_rollouts
        config["gamma"] = 0.995  # discount rate
        config["use_gae"] = True
        config["lambda"] = 0.97
        config["kl_target"] = 0.02
        config["num_sgd_iter"] = 10
        if flags.grid_search:
            config["lambda"] = tune.grid_search([0.5, 0.9])
            config["lr"] = tune.grid_search([5e-4, 5e-5])

        if flags.load_weights_path:
            from flow.controllers.imitation_learning.ppo_model import PPONetwork
            from flow.controllers.imitation_learning.imitation_trainer import Imitation_PPO_Trainable
            from ray.rllib.models import ModelCatalog

            # Register custom model
            ModelCatalog.register_custom_model("PPO_loaded_weights", PPONetwork)
            # set model to the custom model for run
            config['model']['custom_model'] = "PPO_loaded_weights"
            config['model']['custom_options'] = {"h5_load_path": flags.load_weights_path}
            config['observation_filter'] = 'NoFilter'
            # alg run is the Trainable class
            alg_run = Imitation_PPO_Trainable

    elif alg_run == "CENTRALIZEDPPO":
        from flow.algorithms.centralized_PPO import CCTrainer, CentralizedCriticModel
        from ray.rllib.agents.ppo import DEFAULT_CONFIG
        from ray.rllib.models import ModelCatalog
        alg_run = CCTrainer
        config = deepcopy(DEFAULT_CONFIG)
        config['model']['custom_model'] = "cc_model"
        config["model"]["custom_options"]["max_num_agents"] = flow_params['env'].additional_params['max_num_agents']
        config["model"]["custom_options"]["central_vf_size"] = 100

        ModelCatalog.register_custom_model("cc_model", CentralizedCriticModel)

        config["num_workers"] = n_cpus
        config["horizon"] = horizon
        config["model"].update({"fcnet_hiddens": [32, 32]})
        config["train_batch_size"] = horizon * n_rollouts
        config["gamma"] = 0.995  # discount rate
        config["use_gae"] = True
        config["lambda"] = 0.97
        config["kl_target"] = 0.02
        config["num_sgd_iter"] = 10
        if flags.grid_search:
            config["lambda"] = tune.grid_search([0.5, 0.9])
            config["lr"] = tune.grid_search([5e-4, 5e-5])

    elif alg_run == "TD3":
        agent_cls = get_agent_class(alg_run)
        config = deepcopy(agent_cls._default_config)

        config["num_workers"] = n_cpus
        config["horizon"] = horizon
        config["learning_starts"] = 10000
        config["buffer_size"] = 20000  # reduced to test if this is the source of memory problems
        if flags.grid_search:
            config["prioritized_replay"] = tune.grid_search(['True', 'False'])
            config["actor_lr"] = tune.grid_search([1e-3, 1e-4])
            config["critic_lr"] = tune.grid_search([1e-3, 1e-4])
            config["n_step"] = tune.grid_search([1, 10])

    else:
        sys.exit("We only support PPO, TD3, right now.")

    # define some standard and useful callbacks
    def on_episode_start(info):
        episode = info["episode"]
        episode.user_data["avg_speed"] = []
        episode.user_data["avg_speed_avs"] = []
        episode.user_data["avg_energy"] = []
        episode.user_data["avg_mpg"] = []
        episode.user_data["avg_mpj"] = []

    def on_episode_step(info):
        episode = info["episode"]
        env = info["env"].get_unwrapped()[0]
        if isinstance(env, _GroupAgentsWrapper):
            env = env.env
        if hasattr(env, 'no_control_edges'):
            veh_ids = [
                veh_id for veh_id in env.k.vehicle.get_ids()
                if env.k.vehicle.get_speed(veh_id) >= 0
                and env.k.vehicle.get_edge(veh_id) not in env.no_control_edges
            ]
            rl_ids = [
                veh_id for veh_id in env.k.vehicle.get_rl_ids()
                if env.k.vehicle.get_speed(veh_id) >= 0
                and env.k.vehicle.get_edge(veh_id) not in env.no_control_edges
            ]
        else:
            veh_ids = [veh_id for veh_id in env.k.vehicle.get_ids() if env.k.vehicle.get_speed(veh_id) >= 0]
            rl_ids = [veh_id for veh_id in env.k.vehicle.get_rl_ids() if env.k.vehicle.get_speed(veh_id) >= 0]

        speed = np.mean([speed for speed in env.k.vehicle.get_speed(veh_ids)])
        if not np.isnan(speed):
            episode.user_data["avg_speed"].append(speed)
        av_speed = np.mean([speed for speed in env.k.vehicle.get_speed(rl_ids) if speed >= 0])
        if not np.isnan(av_speed):
            episode.user_data["avg_speed_avs"].append(av_speed)
        # episode.user_data["avg_mpg"].append(miles_per_gallon(env, veh_ids, gain=1.0))
        # episode.user_data["avg_mpj"].append(miles_per_megajoule(env, veh_ids, gain=1.0))

    def on_episode_end(info):
        episode = info["episode"]
        avg_speed = np.mean(episode.user_data["avg_speed"])
        episode.custom_metrics["avg_speed"] = avg_speed
        avg_speed_avs = np.mean(episode.user_data["avg_speed_avs"])
        episode.custom_metrics["avg_speed_avs"] = avg_speed_avs
        # episode.custom_metrics["avg_energy_per_veh"] = np.mean(episode.user_data["avg_energy"])
        # episode.custom_metrics["avg_mpg_per_veh"] = np.mean(episode.user_data["avg_mpg"])
        # episode.custom_metrics["avg_mpj_per_veh"] = np.mean(episode.user_data["avg_mpj"])

    def on_train_result(info):
        """Store the mean score of the episode, and adjust the number of adversaries."""
        trainer = info["trainer"]
        # trainer.workers.foreach_worker(
        #     lambda ev: ev.foreach_env(
        #         lambda env: env.set_iteration_num()))

    config["callbacks"] = {"on_episode_start": tune.function(on_episode_start),
                           "on_episode_step": tune.function(on_episode_step),
                           "on_episode_end": tune.function(on_episode_end),
                           "on_train_result": tune.function(on_train_result)}

    # save the flow params for replay
    flow_json = json.dumps(
        flow_params, cls=FlowParamsEncoder, sort_keys=True, indent=4)
    config['env_config']['flow_params'] = flow_json
    config['env_config']['run'] = alg_run

    # multiagent configuration
    if policy_graphs is not None:
        config['multiagent'].update({'policies': policy_graphs})
    if policy_mapping_fn is not None:
        config['multiagent'].update({'policy_mapping_fn': tune.function(policy_mapping_fn)})
    if policies_to_train is not None:
        config['multiagent'].update({'policies_to_train': policies_to_train})

    create_env, gym_name = make_create_env(params=flow_params)

    register_env(gym_name, create_env)
    return alg_run, gym_name, config

def train_rllib(submodule, flags):
    """Train policies using the PPO algorithm in RLlib."""
    import ray
    from ray import tune
    
    class Args:
        def __init__(self):
            self.horizon = 400
            self.algo = 'PPO'
            self.randomize_vehicles = True
    args = Args()
    flow_params = submodule.make_flow_params(args, pedestrians=True)   

    flow_params['sim'].render = flags.render
    policy_graphs = getattr(submodule, "POLICY_GRAPHS", None)
    policy_mapping_fn = getattr(submodule, "policy_mapping_fn", None)
    policies_to_train = getattr(submodule, "policies_to_train", None)

    alg_run, gym_name, config = setup_exps_rllib(
        flow_params, flags.num_cpus, flags.num_rollouts, flags,
        policy_graphs, policy_mapping_fn, policies_to_train)

    config['num_workers'] = flags.num_cpus
    config['env'] = gym_name

    # create a custom string that makes looking at the experiment names easier
    def trial_str_creator(trial):
        return "{}_{}".format(trial.trainable_name, trial.experiment_tag)

    if flags.local_mode:
        print("LOCAL MODE")
        ray.init(local_mode=True)
    else:
        ray.init()

    exp_dict = {
        "run_or_experiment": alg_run,
        "name": flags.exp_title,
        "config": config,
        "checkpoint_freq": flags.checkpoint_freq,
        "checkpoint_at_end": True,
        'trial_name_creator': trial_str_creator,
        "max_failures": 0,
        "stop": {
            "training_iteration": flags.num_iterations,
        },
    }
    date = datetime.now(tz=pytz.utc)
    date = date.astimezone(pytz.timezone('US/Pacific')).strftime("%m-%d-%Y")
    s3_string = "s3://i210.experiments/i210/" \
                + date + '/' + flags.exp_title
    if flags.use_s3:
        exp_dict['upload_dir'] = s3_string
    tune.run(**exp_dict, queue_trials=False, raise_on_failed_trial=False)


def train_h_baselines(flow_params, args, multiagent):
    """Train policies using SAC and TD3 with h-baselines."""
    from hbaselines.algorithms import OffPolicyRLAlgorithm
    from hbaselines.utils.train import parse_options, get_hyperparameters
    from hbaselines.envs.mixed_autonomy import FlowEnv

    flow_params = deepcopy(flow_params)

    # Get the command-line arguments that are relevant here
    args = parse_options(description="", example_usage="", args=args)

    # the base directory that the logged data will be stored in
    base_dir = "training_data"

    # Create the training environment.
    env = FlowEnv(
        flow_params,
        multiagent=multiagent,
        shared=args.shared,
        maddpg=args.maddpg,
        render=args.render,
        version=0
    )

    # Create the evaluation environment.
    if args.evaluate:
        eval_flow_params = deepcopy(flow_params)
        eval_flow_params['env'].evaluate = True
        eval_env = FlowEnv(
            eval_flow_params,
            multiagent=multiagent,
            shared=args.shared,
            maddpg=args.maddpg,
            render=args.render_eval,
            version=1
        )
    else:
        eval_env = None

    for i in range(args.n_training):
        # value of the next seed
        seed = args.seed + i

        # The time when the current experiment started.
        now = strftime("%Y-%m-%d-%H:%M:%S")

        # Create a save directory folder (if it doesn't exist).
        dir_name = os.path.join(base_dir, '{}/{}'.format(args.env_name, now))
        ensure_dir(dir_name)

        # Get the policy class.
        if args.alg == "TD3":
            if multiagent:
                from hbaselines.multi_fcnet.td3 import MultiFeedForwardPolicy
                policy = MultiFeedForwardPolicy
            else:
                from hbaselines.fcnet.td3 import FeedForwardPolicy
                policy = FeedForwardPolicy
        elif args.alg == "SAC":
            if multiagent:
                from hbaselines.multi_fcnet.sac import MultiFeedForwardPolicy
                policy = MultiFeedForwardPolicy
            else:
                from hbaselines.fcnet.sac import FeedForwardPolicy
                policy = FeedForwardPolicy
        else:
            raise ValueError("Unknown algorithm: {}".format(args.alg))

        # Get the hyperparameters.
        hp = get_hyperparameters(args, policy)

        # Add the seed for logging purposes.
        params_with_extra = hp.copy()
        params_with_extra['seed'] = seed
        params_with_extra['env_name'] = args.env_name
        params_with_extra['policy_name'] = policy.__name__
        params_with_extra['algorithm'] = args.alg
        params_with_extra['date/time'] = now

        # Add the hyperparameters to the folder.
        with open(os.path.join(dir_name, 'hyperparameters.json'), 'w') as f:
            json.dump(params_with_extra, f, sort_keys=True, indent=4)

        # Create the algorithm object.
        alg = OffPolicyRLAlgorithm(
            policy=policy,
            env=env,
            eval_env=eval_env,
            **hp
        )

        # Perform training.
        alg.learn(
            total_steps=args.total_steps,
            log_dir=dir_name,
            log_interval=args.log_interval,
            eval_interval=args.eval_interval,
            save_interval=args.save_interval,
            initial_exploration_steps=args.initial_exploration_steps,
            seed=seed,
        )


def train_stable_baselines(submodule, flags):
    """Train policies using the PPO algorithm in stable-baselines."""
    flow_params = submodule.flow_params
    # Path to the saved files
    exp_tag = flow_params['exp_tag']
    result_name = '{}/{}'.format(exp_tag, strftime("%Y-%m-%d-%H:%M:%S"))

    # Perform training.
    print('Beginning training.')
    model = run_model_stablebaseline(
        flow_params, flags.num_cpus, flags.rollout_size, flags.num_steps)

    # Save the model to a desired folder and then delete it to demonstrate
    # loading.
    print('Saving the trained model!')
    path = os.path.realpath(os.path.expanduser('~/baseline_results'))
    ensure_dir(path)
    save_path = os.path.join(path, result_name)
    model.save(save_path)

    # dump the flow params
    with open(os.path.join(path, result_name) + '.json', 'w') as outfile:
        json.dump(flow_params, outfile,
                  cls=FlowParamsEncoder, sort_keys=True, indent=4)

    # Replay the result by loading the model
    print('Loading the trained model and testing it out!')
    model = PPO2.load(save_path)
    flow_params = get_flow_params(os.path.join(path, result_name) + '.json')
    flow_params['sim'].render = True
    env = env_constructor(params=flow_params, version=0)()
    # The algorithms require a vectorized environment to run
    eval_env = DummyVecEnv([lambda: env])
    obs = eval_env.reset()
    reward = 0
    for _ in range(flow_params['env'].horizon):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = eval_env.step(action)
        reward += rewards
    print('the final reward is {}'.format(reward))


def main(args):
    """Perform the training operations."""
    # Parse script-level arguments (not including package arguments).
    flags = parse_args(args)

    # Import relevant information from the exp_config script.
    module = __import__(
        "exp_configs.rl.singleagent", fromlist=[flags.exp_config])
    module_ma = __import__(
        "exp_configs.rl.multiagent", fromlist=[flags.exp_config])

    # Import the sub-module containing the specified exp_config and determine
    # whether the environment is single agent or multi-agent.
    if hasattr(module, flags.exp_config):
        submodule = getattr(module, flags.exp_config)
        multiagent = False
    elif hasattr(module_ma, flags.exp_config):
        submodule = getattr(module_ma, flags.exp_config)
        assert flags.rl_trainer.lower() in ["rllib", "h-baselines"], \
            "Currently, multiagent experiments are only supported through "\
            "RLlib. Try running this experiment using RLlib: " \
            "'python train.py EXP_CONFIG'"
        multiagent = True
    else:
        raise ValueError("Unable to find experiment config.")

    # Perform the training operation.
    if flags.rl_trainer.lower() == "rllib":
        train_rllib(submodule, flags)
    elif flags.rl_trainer.lower() == "stable-baselines":
        train_stable_baselines(submodule, flags)
    elif flags.rl_trainer.lower() == "h-baselines":
        flow_params = submodule.flow_params
        train_h_baselines(flow_params, args, multiagent)
    else:
        raise ValueError("rl_trainer should be either 'rllib', 'h-baselines', "
                         "or 'stable-baselines'.")


if __name__ == "__main__":
    main(sys.argv[1:])
    
    
# """Runner script for single and multi-agent reinforcement learning experiments.

# This script performs an RL experiment using the PPO algorithm. Choice of
# hyperparameters can be seen and adjusted from the code below.

# Usage
#     python train.py EXP_CONFIG
# """
# import argparse
# from datetime import datetime
# import json
# import os
# import sys
# from time import strftime
# from copy import deepcopy

# import numpy as np
# import pytz

# try:
#     from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
#     from stable_baselines import PPO2
# except ImportError:
#     print("Stable-baselines not installed. Please install it if you need it.")

# import ray
# from ray import tune
# from ray.rllib.env.group_agents_wrapper import _GroupAgentsWrapper
# try:
#     from ray.rllib.agents.agent import get_agent_class
# except ImportError:
#     from ray.rllib.agents.registry import get_agent_class
# from ray.tune.registry import register_env

# from flow.core.util import ensure_dir
# # from flow.core.rewards import miles_per_gallon, miles_per_megajoule
# from flow.utils.registry import env_constructor
# from flow.utils.rllib import FlowParamsEncoder, get_flow_params
# from flow.utils.registry import make_create_env


# def parse_args(args):
#     """Parse training options user can specify in command line.

#     Returns
#     -------
#     argparse.Namespace
#         the output parser object
#     """
#     parser = argparse.ArgumentParser(
#         formatter_class=argparse.RawDescriptionHelpFormatter,
#         description="Parse argument used when running a Flow simulation.",
#         epilog="python train.py EXP_CONFIG")

#     # required input parameters
#     parser.add_argument(
#         'exp_config', type=str,
#         help='Name of the experiment configuration file, as located in '
#              'exp_configs/rl/singleagent or exp_configs/rl/multiagent.')

#     parser.add_argument(
#         'exp_title', type=str,
#         help='Name of experiment that results will be stored in')

#     # optional input parameters
#     parser.add_argument(
#         '--rl_trainer', type=str, default="rllib",
#         help='the RL trainer to use. either rllib or Stable-Baselines')
#     parser.add_argument(
#         '--load_weights_path', type=str, default=None,
#         help='Path to h5 file containing a pretrained model. Relevent for PPO with RLLib'
#     )
#     parser.add_argument(
#         '--algorithm', type=str, default="PPO",
#         help='RL algorithm to use. Options are PPO, TD3, and CENTRALIZEDPPO (which uses a centralized value function)'
#              ' right now.'
#     )
#     parser.add_argument(
#         '--num_cpus', type=int, default=1,
#         help='How many CPUs to use')
#     parser.add_argument(
#         '--num_steps', type=int, default=5000,
#         help='How many total steps to perform learning over. Relevant for stable-baselines')
#     parser.add_argument(
#         '--grid_search', action='store_true', default=False,
#         help='Whether to grid search over hyperparams')
#     parser.add_argument(
#         '--num_iterations', type=int, default=200,
#         help='How many iterations are in a training run.')
#     parser.add_argument(
#         '--checkpoint_freq', type=int, default=20,
#         help='How often to checkpoint.')
#     parser.add_argument(
#         '--num_rollouts', type=int, default=1,
#         help='How many rollouts are in a training batch')
#     parser.add_argument(
#         '--rollout_size', type=int, default=1000,
#         help='How many steps are in a training batch.')
#     parser.add_argument('--use_s3', action='store_true', help='If true, upload results to s3')
#     parser.add_argument('--local_mode', action='store_true', default=False,
#                         help='If true only 1 CPU will be used')
#     parser.add_argument('--render', action='store_true', default=False,
#                         help='If true, we render the display')
#     parser.add_argument(
#         '--checkpoint_path', type=str, default=None,
#         help='Directory with checkpoint to restore training from.')

#     return parser.parse_known_args(args)[0]


# def run_model_stablebaseline(flow_params,
#                              num_cpus=1,
#                              rollout_size=50,
#                              num_steps=50):
#     """Run the model for num_steps if provided.

#     Parameters
#     ----------
#     flow_params : dict
#         flow-specific parameters
#     num_cpus : int
#         number of CPUs used during training
#     rollout_size : int
#         length of a single rollout
#     num_steps : int
#         total number of training steps
#     The total rollout length is rollout_size.

#     Returns
#     -------
#     stable_baselines.*
#         the trained model
#     """
#     if num_cpus == 1:
#         constructor = env_constructor(params=flow_params, version=0)()
#         # The algorithms require a vectorized environment to run
#         env = DummyVecEnv([lambda: constructor])
#     else:
#         env = SubprocVecEnv([env_constructor(params=flow_params, version=i)
#                              for i in range(num_cpus)])

#     train_model = PPO2('MlpPolicy', env, verbose=1, n_steps=rollout_size)
#     train_model.learn(total_timesteps=num_steps)
#     return train_model


# def setup_exps_rllib(flow_params,
#                      n_cpus,
#                      n_rollouts,
#                      flags,
#                      policy_graphs=None,
#                      policy_mapping_fn=None,
#                      policies_to_train=None,
#                      ):
#     """Return the relevant components of an RLlib experiment.

#     Parameters
#     ----------
#     flow_params : dict
#         flow-specific parameters (see flow/utils/registry.py)
#     n_cpus : int
#         number of CPUs to run the experiment over
#     n_rollouts : int
#         number of rollouts per training iteration
#     flags:
#         custom arguments
#     policy_graphs : dict, optional
#         TODO
#     policy_mapping_fn : function, optional
#         TODO
#     policies_to_train : list of str, optional
#         TODO
#     Returns
#     -------
#     str
#         name of the training algorithm
#     str
#         name of the gym environment to be trained
#     dict
#         training configuration parameters
#     """

#     horizon = flow_params['env'].horizon

#     alg_run = flags.algorithm.upper()

#     if alg_run == "PPO":
#         from ray import tune
#         from ray.tune.registry import register_env
#         try:
#             from ray.rllib.agents.agent import get_agent_class
#         except ImportError:
#             from ray.rllib.agents.registry import get_agent_class

#         horizon = flow_params['env'].horizon

#         alg_run = "PPO"

#         agent_cls = get_agent_class(alg_run)
#         config = deepcopy(agent_cls._default_config)

#         config["num_workers"] = n_cpus
#         config["horizon"] = horizon
#         config["model"].update({"fcnet_hiddens": [32, 32, 32]})
#         config["train_batch_size"] = horizon * n_rollouts
#         config["gamma"] = 0.995  # discount rate
#         config["use_gae"] = True
#         config["lambda"] = 0.97
#         config["kl_target"] = 0.02
#         config['no_done_at_end'] = True

#         # TODO: restore this to 10
#         config["num_sgd_iter"] = 1
#         # config["num_sgd_iter"] = 10
#         if flags.grid_search:
#             config["lambda"] = tune.grid_search([0.5, 0.9])
#             config["lr"] = tune.grid_search([5e-4, 5e-5])

#         if flags.load_weights_path:
#             from flow.controllers.imitation_learning.ppo_model import PPONetwork
#             from flow.controllers.imitation_learning.imitation_trainer import Imitation_PPO_Trainable
#             from ray.rllib.models import ModelCatalog

#             # Register custom model
#             ModelCatalog.register_custom_model("PPO_loaded_weights", PPONetwork)
#             # set model to the custom model for run
#             config['model']['custom_model'] = "PPO_loaded_weights"
#             config['model']['custom_options'] = {"h5_load_path": flags.load_weights_path}
#             config['observation_filter'] = 'NoFilter'
#             # alg run is the Trainable class 
#             alg_run = Imitation_PPO_Trainable

#     elif alg_run == "CENTRALIZEDPPO":
#         from flow.algorithms.centralized_PPO import CCTrainer, CentralizedCriticModel
#         from ray.rllib.agents.ppo import DEFAULT_CONFIG
#         from ray.rllib.models import ModelCatalog
#         alg_run = CCTrainer
#         config = deepcopy(DEFAULT_CONFIG)
#         config['model']['custom_model'] = "cc_model"
#         config["model"]["custom_options"]["max_num_agents"] = flow_params['env'].additional_params['max_num_agents']
#         config["model"]["custom_options"]["central_vf_size"] = 100

#         ModelCatalog.register_custom_model("cc_model", CentralizedCriticModel)

#         config["num_workers"] = n_cpus
#         config["horizon"] = horizon
#         config["model"].update({"fcnet_hiddens": [32, 32]})
#         config["train_batch_size"] = horizon * n_rollouts
#         config["gamma"] = 0.995  # discount rate
#         config["use_gae"] = True
#         config["lambda"] = 0.97
#         config["kl_target"] = 0.02
#         config["num_sgd_iter"] = 10
#         if flags.grid_search:
#             config["lambda"] = tune.grid_search([0.5, 0.9])
#             config["lr"] = tune.grid_search([5e-4, 5e-5])

#     elif alg_run == "TD3":
#         agent_cls = get_agent_class(alg_run)
#         config = deepcopy(agent_cls._default_config)

#         config["num_workers"] = n_cpus
#         config["horizon"] = horizon
#         config["learning_starts"] = 10000
#         config["buffer_size"] = 20000  # reduced to test if this is the source of memory problems
#         if flags.grid_search:
#             config["prioritized_replay"] = tune.grid_search(['True', 'False'])
#             config["actor_lr"] = tune.grid_search([1e-3, 1e-4])
#             config["critic_lr"] = tune.grid_search([1e-3, 1e-4])
#             config["n_step"] = tune.grid_search([1, 10])

#     else:
#         sys.exit("We only support PPO, TD3, right now.")

#     # define some standard and useful callbacks
#     def on_episode_start(info):
#         episode = info["episode"]
#         episode.user_data["avg_speed"] = []
#         episode.user_data["avg_speed_avs"] = []
#         episode.user_data["avg_energy"] = []
#         episode.user_data["avg_mpg"] = []
#         episode.user_data["avg_mpj"] = []


#     def on_episode_step(info):
#         episode = info["episode"]
#         env = info["env"].get_unwrapped()[0]
#         if isinstance(env, _GroupAgentsWrapper):
#             env = env.env
#         if hasattr(env, 'no_control_edges'):
#             veh_ids = [veh_id for veh_id in env.k.vehicle.get_ids() if (env.k.vehicle.get_speed(veh_id) >= 0
#                                                                         and env.k.vehicle.get_edge(veh_id)
#                                                                         not in env.no_control_edges)]
#             rl_ids = [veh_id for veh_id in env.k.vehicle.get_rl_ids() if (env.k.vehicle.get_speed(veh_id) >= 0
#                                                                         and env.k.vehicle.get_edge(veh_id)
#                                                                         not in env.no_control_edges)]
#         else:
#             veh_ids = [veh_id for veh_id in env.k.vehicle.get_ids() if env.k.vehicle.get_speed(veh_id) >= 0]
#             rl_ids = [veh_id for veh_id in env.k.vehicle.get_rl_ids() if env.k.vehicle.get_speed(veh_id) >= 0]

#         speed = np.mean([speed for speed in env.k.vehicle.get_speed(veh_ids)])
#         if not np.isnan(speed):
#             episode.user_data["avg_speed"].append(speed)
#         av_speed = np.mean([speed for speed in env.k.vehicle.get_speed(rl_ids) if speed >= 0])
#         if not np.isnan(av_speed):
#             episode.user_data["avg_speed_avs"].append(av_speed)


#     def on_episode_end(info):
#         episode = info["episode"]
#         avg_speed = np.mean(episode.user_data["avg_speed"])
#         episode.custom_metrics["avg_speed"] = avg_speed
#         avg_speed_avs = np.mean(episode.user_data["avg_speed_avs"])
#         episode.custom_metrics["avg_speed_avs"] = avg_speed_avs

#     def on_train_result(info):
#         """Store the mean score of the episode, and increment or decrement how many adversaries are on"""
#         trainer = info["trainer"]
#         # trainer.workers.foreach_worker(
#         #     lambda ev: ev.foreach_env(
#         #         lambda env: env.set_iteration_num()))

#     config["callbacks"] = {"on_episode_start": tune.function(on_episode_start),
#                            "on_episode_step": tune.function(on_episode_step),
#                            "on_episode_end": tune.function(on_episode_end),
#                            "on_train_result": tune.function(on_train_result)}

#     # save the flow params for replay
#     flow_json = json.dumps(
#         flow_params, cls=FlowParamsEncoder, sort_keys=True, indent=4)
#     config['env_config']['flow_params'] = flow_json
#     config['env_config']['run'] = alg_run

#     # multiagent configuration
#     if policy_graphs is not None:
#         config['multiagent'].update({'policies': policy_graphs})
#     if policy_mapping_fn is not None:
#         config['multiagent'].update({'policy_mapping_fn': tune.function(policy_mapping_fn)})
#     if policies_to_train is not None:
#         config['multiagent'].update({'policies_to_train': policies_to_train})

#     create_env, gym_name = make_create_env(params=flow_params)

#     register_env(gym_name, create_env)
#     return alg_run, gym_name, config

# # def train_rllib_with_imitation(submodule, flags):
# #     """Train policies using the PPO algorithm in RLlib, with initiale policy weights from imitation learning."""
# #     import ray
# #     from flow.controllers.imitation_learning.ppo_model import PPONetwork
# #     from ray.rllib.models import ModelCatalog
# #
# #     flow_params = submodule.flow_params
# #     flow_params['sim'].render = flags.render
# #     policy_graphs = getattr(submodule, "POLICY_GRAPHS", None)
# #     policy_mapping_fn = getattr(submodule, "policy_mapping_fn", None)
# #     policies_to_train = getattr(submodule, "policies_to_train", None)
# #
# #     alg_run, gym_name, config = setup_exps_rllib(
# #         flow_params, flags.num_cpus, flags.num_rollouts, flags,
# #         policy_graphs, policy_mapping_fn, policies_to_train)
# #
# #
# #
# #     config['num_workers'] = flags.num_cpus
# #     config['env'] = gym_name
# #
# #     # create a custom string that makes looking at the experiment names easier
# #     def trial_str_creator(trial):
# #         return "{}_{}".format(trial.trainable_name, trial.experiment_tag)
# #
# #     if flags.local_mode:
# #         ray.init(local_mode=True)
# #     else:
# #         ray.init()
# #
# #     exp_dict = {
# #         "run_or_experiment": alg_run,
# #         "name": gym_name,
# #         "config": config,
# #         "checkpoint_freq": flags.checkpoint_freq,
# #         "checkpoint_at_end": True,
# #         'trial_name_creator': trial_str_creator,
# #         "max_failures": 0,
# #         "stop": {
# #             "training_iteration": flags.num_iterations,
# #         },
# #     }
# #     date = datetime.now(tz=pytz.utc)
# #     date = date.astimezone(pytz.timezone('US/Pacific')).strftime("%m-%d-%Y")
# #     s3_string = "s3://i210.experiments/i210/" \
# #                 + date + '/' + flags.exp_title
# #     if flags.use_s3:
# #         exp_dict['upload_dir'] = s3_string
# #     tune.run(**exp_dict, queue_trials=False, raise_on_failed_trial=False)

# def train_rllib(submodule, flags):
#     """Train policies using the PPO algorithm in RLlib."""
#     class Args:
#         def __init__(self):
#             self.horizon = 400
#             self.algo = 'PPO'
#     args = Args()
#     flow_params = submodule.make_flow_params(args, pedestrians=True)    
#     flow_params['sim'].render = flags.render
#     policy_graphs = getattr(submodule, "POLICY_GRAPHS", None)
#     policy_mapping_fn = getattr(submodule, "policy_mapping_fn", None)
#     policies_to_train = getattr(submodule, "policies_to_train", None)

#     alg_run, gym_name, config = setup_exps_rllib(
#         flow_params, flags.num_cpus, flags.num_rollouts, flags,
#         policy_graphs, policy_mapping_fn, policies_to_train)

#     config['num_workers'] = flags.num_cpus
#     config['env'] = gym_name

#     # create a custom string that makes looking at the experiment names easier
#     def trial_str_creator(trial):
#         return "{}_{}".format(trial.trainable_name, trial.experiment_tag)

#     if flags.local_mode:
#         print("LOCAL MODE")
#         ray.init(local_mode=True)
#     else:
#         ray.init()

#     exp_dict = {
#         "run_or_experiment": alg_run,
#         "name": flags.exp_title,
#         "config": config,
#         "checkpoint_freq": flags.checkpoint_freq,
#         "checkpoint_at_end": True,
#         'trial_name_creator': trial_str_creator,
#         "max_failures": 0,
#         "stop": {
#             "training_iteration": flags.num_iterations,
#         },
#     }
#     date = datetime.now(tz=pytz.utc)
#     date = date.astimezone(pytz.timezone('US/Pacific')).strftime("%m-%d-%Y")
#     s3_string = "s3://i210.experiments/i210/" \
#                 + date + '/' + flags.exp_title
#     if flags.use_s3:
#         exp_dict['upload_dir'] = s3_string
#     tune.run(**exp_dict, queue_trials=False, raise_on_failed_trial=False)


# def train_h_baselines(flow_params, args, multiagent):
#     """Train policies using SAC and TD3 with h-baselines."""
#     from hbaselines.algorithms import OffPolicyRLAlgorithm
#     from hbaselines.utils.train import parse_options, get_hyperparameters
#     from hbaselines.envs.mixed_autonomy import FlowEnv

#     flow_params = deepcopy(flow_params)

#     # Get the command-line arguments that are relevant here
#     args = parse_options(description="", example_usage="", args=args)

#     # the base directory that the logged data will be stored in
#     base_dir = "training_data"

#     # Create the training environment.
#     env = FlowEnv(
#         flow_params,
#         multiagent=multiagent,
#         shared=args.shared,
#         maddpg=args.maddpg,
#         render=args.render,
#         version=0
#     )

#     # Create the evaluation environment.
#     if args.evaluate:
#         eval_flow_params = deepcopy(flow_params)
#         eval_flow_params['env'].evaluate = True
#         eval_env = FlowEnv(
#             eval_flow_params,
#             multiagent=multiagent,
#             shared=args.shared,
#             maddpg=args.maddpg,
#             render=args.render_eval,
#             version=1
#         )
#     else:
#         eval_env = None

#     for i in range(args.n_training):
#         # value of the next seed
#         seed = args.seed + i

#         # The time when the current experiment started.
#         now = strftime("%Y-%m-%d-%H:%M:%S")

#         # Create a save directory folder (if it doesn't exist).
#         dir_name = os.path.join(base_dir, '{}/{}'.format(args.env_name, now))
#         ensure_dir(dir_name)

#         # Get the policy class.
#         if args.alg == "TD3":
#             if multiagent:
#                 from hbaselines.multi_fcnet.td3 import MultiFeedForwardPolicy
#                 policy = MultiFeedForwardPolicy
#             else:
#                 from hbaselines.fcnet.td3 import FeedForwardPolicy
#                 policy = FeedForwardPolicy
#         elif args.alg == "SAC":
#             if multiagent:
#                 from hbaselines.multi_fcnet.sac import MultiFeedForwardPolicy
#                 policy = MultiFeedForwardPolicy
#             else:
#                 from hbaselines.fcnet.sac import FeedForwardPolicy
#                 policy = FeedForwardPolicy
#         else:
#             raise ValueError("Unknown algorithm: {}".format(args.alg))

#         # Get the hyperparameters.
#         hp = get_hyperparameters(args, policy)

#         # Add the seed for logging purposes.
#         params_with_extra = hp.copy()
#         params_with_extra['seed'] = seed
#         params_with_extra['env_name'] = args.env_name
#         params_with_extra['policy_name'] = policy.__name__
#         params_with_extra['algorithm'] = args.alg
#         params_with_extra['date/time'] = now

#         # Add the hyperparameters to the folder.
#         with open(os.path.join(dir_name, 'hyperparameters.json'), 'w') as f:
#             json.dump(params_with_extra, f, sort_keys=True, indent=4)

#         # Create the algorithm object.
#         alg = OffPolicyRLAlgorithm(
#             policy=policy,
#             env=env,
#             eval_env=eval_env,
#             **hp
#         )

#         # Perform training.
#         alg.learn(
#             total_timesteps=args.total_steps,
#             log_dir=dir_name,
#             log_interval=args.log_interval,
#             eval_interval=args.eval_interval,
#             save_interval=args.save_interval,
#             initial_exploration_steps=args.initial_exploration_steps,
#             seed=seed,
#         )


# def train_stable_baselines(submodule, flags):
#     """Train policies using the PPO algorithm in stable-baselines."""
#     flow_params = submodule.flow_params
#     # Path to the saved files
#     exp_tag = flow_params['exp_tag']
#     result_name = '{}/{}'.format(exp_tag, strftime("%Y-%m-%d-%H:%M:%S"))

#     # Perform training.
#     print('Beginning training.')
#     model = run_model_stablebaseline(
#         flow_params, flags.num_cpus, flags.rollout_size, flags.num_steps)

#     # Save the model to a desired folder and then delete it to demonstrate
#     # loading.
#     print('Saving the trained model!')
#     path = os.path.realpath(os.path.expanduser('~/baseline_results'))
#     ensure_dir(path)
#     save_path = os.path.join(path, result_name)
#     model.save(save_path)

#     # dump the flow params
#     with open(os.path.join(path, result_name) + '.json', 'w') as outfile:
#         json.dump(flow_params, outfile,
#                   cls=FlowParamsEncoder, sort_keys=True, indent=4)

#     # Replay the result by loading the model
#     print('Loading the trained model and testing it out!')
#     model = PPO2.load(save_path)
#     flow_params = get_flow_params(os.path.join(path, result_name) + '.json')
#     flow_params['sim'].render = True
#     env = env_constructor(params=flow_params, version=0)()
#     # The algorithms require a vectorized environment to run
#     eval_env = DummyVecEnv([lambda: env])
#     obs = eval_env.reset()
#     reward = 0
#     for _ in range(flow_params['env'].horizon):
#         action, _states = model.predict(obs)
#         obs, rewards, dones, info = eval_env.step(action)
#         reward += rewards
#     print('the final reward is {}'.format(reward))


# def main(args):
#     """Perform the training operations."""
#     # Parse script-level arguments (not including package arguments).
#     flags = parse_args(args)

#     # Import relevant information from the exp_config script.
#     module = __import__(
#         "exp_configs.rl.singleagent", fromlist=[flags.exp_config])
#     module_ma = __import__(
#         "exp_configs.rl.multiagent", fromlist=[flags.exp_config])

#     # Import the sub-module containing the specified exp_config and determine
#     # whether the environment is single agent or multi-agent.
#     if hasattr(module, flags.exp_config):
#         submodule = getattr(module, flags.exp_config)
#         multiagent = False
#     elif hasattr(module_ma, flags.exp_config):
#         submodule = getattr(module_ma, flags.exp_config)
#         assert flags.rl_trainer.lower() in ["rllib", "h-baselines"], \
#             "Currently, multiagent experiments are only supported through "\
#             "RLlib. Try running this experiment using RLlib: " \
#             "'python train.py EXP_CONFIG'"
#         multiagent = True
#     else:
#         raise ValueError("Unable to find experiment config.")

#     # Perform the training operation.
#     if flags.rl_trainer.lower() == "rllib":
#         train_rllib(submodule, flags)
#     elif flags.rl_trainer.lower() == "stable-baselines":
#         train_stable_baselines(submodule, flags)
#     elif flags.rl_trainer.lower() == "h-baselines":
#         flow_params = submodule.flow_params
#         train_h_baselines(flow_params, args, multiagent)
#     else:
#         raise ValueError("rl_trainer should be either 'rllib', 'h-baselines', "
#                          "or 'stable-baselines'.")


