"""There is a vehicle obscuring a pedestrian that conflicts with your path."""

import os
import ray

from flow.controllers import GridRouter, PreTrainedController
from flow.core.experiment import Experiment
from flow.core.bayesian_0_experiment import Bayesian0Experiment
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams, SumoLaneChangeParams
from flow.core.params import VehicleParams
from flow.core.params import SumoCarFollowingParams
from flow.envs.multiagent.bayesian_0_no_grid_env import Bayesian0NoGridEnv, ADDITIONAL_ENV_PARAMS
from flow.networks import Bayesian0Network
from flow.core.params import PedestrianParams

from flow.utils.registry import make_create_env
from flow.utils.rllib import FlowParamsEncoder

# Experiment parameters
N_ROLLOUTS = 20  # number of rollouts per training iteration
N_CPUS = 8 # number of parallel workers

# Environment parameters
# TODO(@klin) make sure these parameters match what you've set up in the SUMO version here
V_ENTER = 30  # enter speed for departing vehicles
INNER_LENGTH = 50  # length of inner edges in the traffic light grid network
# number of vehicles originating in the left, right, top, and bottom edges
N_LEFT, N_RIGHT, N_TOP, N_BOTTOM = 0, 1, 1, 1


def get_flow_params(args=None, pedestrians=True, render=True):
    pedestrian_params = None
    if pedestrians:
        pedestrian_params = PedestrianParams()
        # pedestrian_params.add(
        #     ped_id='ped_0',
        #     depart_time='0.00',
        #     start='(1.0)--(1.1)',
        #     end='(1.1)--(1.2)',
        #     depart_pos='40',
        #     )
        for i in range(20):
            pedestrian_params.add(
                ped_id=f'ped_{i}',
                depart_time='0.00',
                start='(1.2)--(1.1)',
                end='(1.1)--(1.0)',
                depart_pos=f'{44 + 0.5 * i}',
                arrival_pos='5')

    # we place a sufficient number of vehicles to ensure they confirm with the
    # total number specified above. We also use a "right_of_way" speed mode to
    # support traffic light compliance

    vehicles = VehicleParams()

    vehicles.add(
        veh_id="av",
        routing_controller=(GridRouter, {}),
        car_following_params=SumoCarFollowingParams(
            min_gap=2.5,
            decel=7.5,  # avoid collisions at emergency stops
            speed_mode="aggressive",
        ),
        color='white',
        acceleration_controller=(PreTrainedController,
                                 {"path": os.path.expanduser("~/ray_results/final_policy_rss/DQN_0_0_2020-06-24_14-19-463mwnbpq0"),
                                  "checkpoint_num": str(400)}),
        num_vehicles=2)

    vehicles.add(
        veh_id="rl",
        routing_controller=(GridRouter, {}),
        car_following_params=SumoCarFollowingParams(
            min_gap=2.5,
            decel=7.5,  # avoid collisions at emergency stops
            speed_mode="aggressive",
        ),
        color='red',
        acceleration_controller=(PreTrainedController,
                                 {"path": os.path.expanduser("~/ray_results/final_policy_rss/DQN_0_0_2020-06-24_14-19-463mwnbpq0"),
                                  "checkpoint_num": str(400)}),
        num_vehicles=1)

    n_rows = 1
    n_columns = 1

    # define initial configs to pass into dict
    initial_config = InitialConfig(
        spacing='custom',
        shuffle=False,
        sidewalks=True,
        lanes_distribution=float('inf'))

    net_params = NetParams(
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
            }
    )

    flow_params = dict(
        # name of the experiment
        exp_tag="bayesian_0_env",

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

        env= EnvParams(
        horizon=500,
        # environment related parameters (see flow.core.params.EnvParams)
        additional_params=ADDITIONAL_ENV_PARAMS,
        ),
        # network-related parameters (see flow.core.params.NetParams and the
        # network's documentation or ADDITIONAL_NET_PARAMS component)
        net=net_params,
        # vehicles to be placed in the network at the start of a rollout (see
        # flow.core.params.VehicleParams)
        veh=vehicles,

        ped=pedestrian_params,

        # parameters specifying the positioning of vehicles upon initialization
        # or reset (see flow.core.params.InitialConfig)
        initial=initial_config,
        network_init=Bayesian0Network(
            name="bayesian_1",
            vehicles=vehicles,
            net_params=net_params,
            pedestrians=pedestrian_params,
            initial_config=initial_config)
    )
    return flow_params


def make_env(args=None, bayesian_0_param=None):
    """
    Generate the flow params for the experiment.
    Parameters
    ----------
    Returns
    -------
    env
    """

    flow_params = get_flow_params()
    env = Bayesian0NoGridEnv(flow_params['env'], flow_params['sim'], flow_params['network_init'])

    return env, 'Bayesian0NoGridEnv-v0'

if __name__ == '__main__':
    env, name = make_env()
    exp = Experiment(env)
    ray.init()
    exp.run(4, 1500, multiagent=True)