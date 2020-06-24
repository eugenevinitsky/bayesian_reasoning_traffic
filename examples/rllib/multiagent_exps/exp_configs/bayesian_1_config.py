"""There is a vehicle obscuring a pedestrian that conflicts with your path."""

from flow.envs.multiagent import Bayesian0NoGridEnv
from flow.networks import Bayesian1Network
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
# TODO(@klin) make sure these parameters match what you've set up in the SUMO version here
V_ENTER = 30  # enter speed for departing vehicles
INNER_LENGTH = 50  # length of inner edges in the traffic light grid network
# number of vehicles originating in the left, right, top, and bottom edges
N_LEFT, N_RIGHT, N_TOP, N_BOTTOM = 0, 1, 1, 1


def make_flow_params():
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
    pedestrian_params.add(
        ped_id='ped_0',
        depart_time='0.00',
        start='(1.0)--(1.1)',
        end='(1.1)--(1.2)',
        depart_pos='40')

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
        num_vehicles=2)

    vehicles.add(
        veh_id='rl',
        acceleration_controller=(RLController, {}),
        car_following_params=SumoCarFollowingParams(
            speed_mode="right_of_way",
        ),
        routing_controller=(GridRouter, {}),
        num_vehicles=1)

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
    initial_config = InitialConfig(
        spacing='custom',
        shuffle=False,
        sidewalks=True,
        lanes_distribution=float('inf'))

    flow_params = dict(
        # name of the experiment
        exp_tag="bayesian_1_env",

        # name of the flow environment the experiment is running on
        env_name=Bayesian0NoGridEnv,

        # name of the network class the experiment is running on
        network=Bayesian1Network,

        # simulator that is used by the experiment
        simulator='traci',

        # sumo-related parameters (see flow.core.params.SumoParams)
        sim=SumoParams(
            restart_instance=False,
            sim_step=0.1,
            render=False,
        ),

        env=EnvParams(
            horizon=500,
            # environment related parameters (see flow.core.params.EnvParams)
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
                "discrete": False,
                # whether to randomize which edge the vehicles are coming from
                "randomize_vehicles": False,
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
