"""There is a line of stopped vehicles; we need to infer why they are stopped."""

from flow.envs.ring.accel import AccelEnv, AccelWithQueryEnv, ADDITIONAL_ENV_PARAMS
from flow.controllers.bayesian_predict_controller import BayesianPredictController
from flow.networks import Bayesian3Network
from flow.controllers.velocity_controllers import FullStop
from flow.controllers import GridRouter
from flow.core.params import SumoCarFollowingParams, VehicleParams
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams, PedestrianParams


def make_env():
    """
    Generate the flow params for the experiment.

    Parameters
    ----------
    n_rows : int
        number of rows in the traffic light grid
    n_columns : int
        number of columns in the traffic light grid
    edge_inflow : float


    Returns
    -------
    dict
        flow_params object
    """
    # we place a sufficient number of vehicles to ensure they confirm with the
    # total number specified above. We also use a "right_of_way" speed mode to
    # support traffic light compliance
    v_enter = 10
    inner_length = 50
    n_rows = 1
    n_columns = 1
    # TODO(@nliu) add the pedestrian in
    num_cars_left = 0
    num_cars_right = 0
    num_cars_top = 0
    num_cars_bot = 1

    grid_array = {
        "inner_length": inner_length,
        "row_num": n_rows,
        "col_num": n_columns,
        "cars_left": num_cars_left,
        "cars_right": num_cars_right,
        "cars_top": num_cars_top,
        "cars_bot": num_cars_bot
    }

    # we don't need any pedestrians since we're going to fake one
    pedestrian_params = PedestrianParams()
    vehicles = VehicleParams()

    # this will be the l2 vehicle
    vehicles.add(
        veh_id="l2",
        acceleration_controller=(BayesianPredictController, {}),
        routing_controller=(GridRouter, {}),
        car_following_params=SumoCarFollowingParams(
            min_gap=2.5,
            decel=7.5,  # avoid collisions at emergency stops
            speed_mode="aggressive",
        ),
        color='red',
        num_vehicles=1)

    # this will be the l1 vehicle
    vehicles.add(
        veh_id="av",
        routing_controller=(GridRouter, {}),
        car_following_params=SumoCarFollowingParams(
            min_gap=2.5,
            decel=7.5,  # avoid collisions at emergency stops
            speed_mode="aggressive",
        ),
        num_vehicles=1)

    additional_net_params = {
        "grid_array": grid_array,
        "speed_limit": 35,
        "horizontal_lanes": 2,
        "vertical_lanes": 2
    }

    initial = InitialConfig(
        spacing='custom', sidewalks=True, lanes_distribution=float('inf'), shuffle=False)
    net = NetParams(additional_params=additional_net_params)

    net = Bayesian3Network(
        name="bayesian_3",
        vehicles=vehicles,
        net_params=net,
        pedestrians=pedestrian_params,
        initial_config=initial)

    flow_params = dict(
        # name of the experiment
        exp_tag="why_are_they_stopped",

        # name of the network class the experiment is running on
        network=Bayesian3Network,

        # simulator that is used by the experiment
        simulator='traci',

        # sumo-related parameters (see flow.core.params.SumoParams)
        sim=SumoParams(sim_step=0.1,
                       render=False,
                       restart_instance=False),

        # environment related parameters (see flow.core.params.EnvParams)
        env=EnvParams(
            horizon=500,
            additional_params=ADDITIONAL_ENV_PARAMS,
        ),

        # network-related parameters (see flow.core.params.NetParams and the
        # network's documentation or ADDITIONAL_NET_PARAMS component)
        net=net,

        # vehicles to be placed in the network at the start of a rollout (see
        # flow.core.params.VehicleParams)
        veh=vehicles,

        ped=pedestrian_params,

        # parameters specifying the positioning of vehicles upon initialization
        # or reset (see flow.core.params.InitialConfig)
        initial=initial,
    )

    env = AccelWithQueryEnv(flow_params['env'], flow_params['sim'], flow_params['net'])
    env.query_env = AccelEnv(flow_params['env'], flow_params['sim'], flow_params['net'])
    return env