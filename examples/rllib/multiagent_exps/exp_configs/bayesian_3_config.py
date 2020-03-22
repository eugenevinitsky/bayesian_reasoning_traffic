"""Check if the AV learns to slow down and not hit a pedestrian that is invisible."""

from flow.envs.multiagent import Bayesian1Env
from flow.networks import Bayesian3Network
from flow.controllers import GridRouter
from flow.core.params import SumoCarFollowingParams, VehicleParams
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams, PedestrianParams
from flow.envs.multiagent.bayesian_1_env import Bayesian1Env, ADDITIONAL_ENV_PARAMS


def get_non_flow_params(enter_speed, add_net_params, pedestrians=False):
    """Define the network and initial params in the absence of inflows.

    Parameters
    ----------
    enter_speed : float
        initial speed of vehicles as they enter the network.
    add_net_params: dict
        additional network-specific parameters (unique to the traffic light grid)

    Returns
    -------
    flow.core.params.InitialConfig
        parameters specifying the initial configuration of vehicles in the
        network
    flow.core.params.NetParams
        network-specific parameters used to generate the network
    """
    additional_init_params = {'enter_speed': enter_speed}
    initial = InitialConfig(
        spacing='custom', additional_params=additional_init_params, shuffle=False)
    if pedestrians:
        initial = InitialConfig(
            spacing='custom', sidewalks=True, lanes_distribution=float('inf'), shuffle=False)
    net = NetParams(additional_params=add_net_params)

    return initial, net

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
    pedestrian_params.add(
        ped_id='ped_1',
        depart_time='0.00',
        start='(1.2)--(1.1)',
        end='(2.1)--(1.1)',
        depart_pos='45')

    vehicles = VehicleParams()

    vehicles.add(
        veh_id="human",
        routing_controller=(GridRouter, {}),
        car_following_params=SumoCarFollowingParams(
            min_gap=2.5,
            decel=7.5,  # avoid collisions at emergency stops
            speed_mode="right_of_way",
        ),
        num_vehicles=1)

    vehicles.add(
        veh_id="obstacle",
        routing_controller=(GridRouter, {}),
        car_following_params=SumoCarFollowingParams(
            min_gap=2.5,
            decel=7.5,  # avoid collisions at emergency stops
            speed_mode="right_of_way",
            max_speed=0.0000000000001
        ),
        num_vehicles=3)

    additional_net_params = {
        "grid_array": grid_array,
        "speed_limit": 35,
        "horizontal_lanes": 2,
        "vertical_lanes": 2
    }

    initial_config, net_params = get_non_flow_params(
        enter_speed=v_enter,
        add_net_params=additional_net_params,
        pedestrians=True)

    flow_params = dict(
        # name of the experiment
        exp_tag="why_are_they_stopped",

        # name of the flow environment the experiment is running on
        env_name=Bayesian1Env,

        # name of the network class the experiment is running on
        network=Bayesian3Network,

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
        net=net_params,

        # vehicles to be placed in the network at the start of a rollout (see
        # flow.core.params.VehicleParams)
        veh=vehicles,

        ped=pedestrian_params,

        # parameters specifying the positioning of vehicles upon initialization
        # or reset (see flow.core.params.InitialConfig)
        initial=InitialConfig(
            spacing='custom',
            shuffle=False,
        ),
    )
    return flow_params
