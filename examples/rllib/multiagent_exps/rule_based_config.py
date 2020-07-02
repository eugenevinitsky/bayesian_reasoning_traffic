from flow.envs.multiagent import Bayesian0NoGridEnv
from flow.networks import Bayesian0Network
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.params import SumoCarFollowingParams, VehicleParams
from flow.core.params import PedestrianParams

from flow.controllers import SimCarFollowingController, GridRouter, RuleBasedIntersectionController
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
            depart_pos=f'{44 + 0.5 * i}',
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

        # TODO(klin) make sure the autonomous vehicle being placed here is placed in the right position
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
        initial=initial_config
    )

    return flow_params
