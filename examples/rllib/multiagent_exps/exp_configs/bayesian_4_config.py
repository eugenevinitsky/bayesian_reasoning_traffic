"""Check if the AV learns to slow down and not hit a pedestrian that is invisible."""

from flow.networks import Bayesian4Network
from flow.controllers.velocity_controllers import FullStop
from flow.controllers import GridRouter, RLController
from flow.core.params import SumoCarFollowingParams, VehicleParams
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams, PedestrianParams
from flow.envs.multiagent.bayesian_1_env import Bayesian1Env, ADDITIONAL_ENV_PARAMS


def make_flow_params():
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
    num_cars_bot = 4

    grid_array = {
        "inner_length": inner_length,
        "row_num": n_rows,
        "col_num": n_columns,
        "cars_left": num_cars_left,
        "cars_right": num_cars_right,
        "cars_top": num_cars_top,
        "cars_bot": num_cars_bot
    }

    pedestrian_params = PedestrianParams()
    pedestrian_params.add(
        ped_id='ped_0',
        depart_time='0.00',
        start='(1.2)--(1.1)',
        end='(2.1)--(1.1)',
        depart_pos='43')

    vehicles = VehicleParams()

    vehicles.add(
        veh_id="obstacle",
        routing_controller=(GridRouter, {}),
        car_following_params=SumoCarFollowingParams(
            min_gap=2.5,
            decel=7.5,  # avoid collisions at emergency stops
            speed_mode="right_of_way",
            max_speed=0.0000000000001
        ),
        acceleration_controller=(FullStop, {}),
        num_vehicles=1)

    vehicles.add(
        veh_id="av",
        routing_controller=(GridRouter, {}),
        car_following_params=SumoCarFollowingParams(
            min_gap=2.5,
            decel=7.5,  # avoid collisions at emergency stops
            speed_mode="right_of_way",
        ),
        acceleration_controller=(RLController, {}),
        num_vehicles=1)

    additional_net_params = {
        "grid_array": grid_array,
        "speed_limit": 35,
        "horizontal_lanes": 1,
        "vertical_lanes": 1
    }

    initial = InitialConfig(
        spacing='custom', sidewalks=True, lanes_distribution=float('inf'), shuffle=False)
    net = NetParams(additional_params=additional_net_params)

    flow_params = dict(
        # name of the experiment
        exp_tag="hidden_pedestrian",

        # name of the flow environment the experiment is running on
        env_name=Bayesian1Env,

        # name of the network class the experiment is running on
        network=Bayesian4Network,

        # simulator that is used by the experiment
        simulator='traci',

        # sumo-related parameters (see flow.core.params.SumoParams)
        sim=SumoParams(sim_step=0.1,
                       render=False,
                       restart_instance=True),

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
    return flow_params
