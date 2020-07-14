"""Sets up and runs the basic bayesian example. This script is just for debugging and checking that everything
actually arrives at the desired time so that the conflict occurs. """

import argparse
import os
import ray

from flow.controllers import GridRouter, RuleBasedIntersectionController
from flow.core.experiment import Experiment
from flow.core.bayesian_0_experiment import Bayesian0Experiment
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams, SumoLaneChangeParams
from flow.core.params import VehicleParams
from flow.core.params import SumoCarFollowingParams
from flow.envs.multiagent.bayesian_0_no_grid_env import Bayesian0NoGridEnv, ADDITIONAL_ENV_PARAMS
from flow.networks import Bayesian0Network
from flow.core.params import PedestrianParams


def gen_edges(col_num, row_num):
    """Generate the names of the outer edges in the traffic light grid network.

    Parameters
    ----------
    col_num : int
        number of columns in the traffic light grid
    row_num : int
        number of rows in the traffic light grid

    Returns
    -------
    list of str
        names of all the outer edges
    """
    edges = []

    x_max = col_num + 1
    y_max = row_num + 1

    def new_edge(from_node, to_node):
        return str(from_node) + "--" + str(to_node)

    # Build the horizontal edges
    for y in range(1, y_max):
        for x in [0, x_max - 1]:
            left_node = "({}.{})".format(x, y)
            right_node = "({}.{})".format(x + 1, y)
            edges += new_edge(left_node, right_node)
            edges += new_edge(right_node, left_node)

    # Build the vertical edges
    for x in range(1, x_max):
        for y in [0, y_max - 1]:
            bottom_node = "({}.{})".format(x, y)
            top_node = "({}.{})".format(x, y + 1)
            edges += new_edge(bottom_node, top_node)
            edges += new_edge(top_node, bottom_node)

    return edges


def get_non_flow_params(enter_speed, add_net_params, pedestrians=False):
    """Define the network and initial params in the absence of inflows.

    Note that when a vehicle leaves a network in this case, it is immediately #TODO(KLin) Does this actually happen?
    returns to the start of the row/column it was traversing, and in the same
    direction as it was before.

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


def bayesian_0_example(render=None, pedestrians=False, collect_data=False):
    """
    Perform a simulation of vehicles on a traffic light grid.

    Parameters
    ----------
    render: bool, optional
        specifies whether to use the gui during execution

    Returns
    -------
    exp: flow.core.experiment.Experiment
        A non-rl experiment demonstrating the performance of human-driven
        vehicles and balanced traffic lights on a traffic light grid.
    """
    v_enter = 10
    inner_length = 50
    n_rows = 1
    n_columns = 1
    num_cars_left = 1
    num_cars_right = 1
    num_cars_top = 1
    num_cars_bot = 0
    tot_cars = (num_cars_left + num_cars_right) * n_columns \
               + (num_cars_top + num_cars_bot) * n_rows  # Why's this * n_rows and not n_cols?

    grid_array = {
        "inner_length": inner_length,
        "row_num": n_rows,
        "col_num": n_columns,
        "cars_left": num_cars_left,
        "cars_right": num_cars_right,
        "cars_top": num_cars_top,
        "cars_bot": num_cars_bot
    }

    sim_params = SumoParams(sim_step=1.0, render=True, restart_instance=True)

    if render is not None:
        sim_params.render = render

    lane_change_params = SumoLaneChangeParams(
        lc_assertive=20,
        lc_pushy=0.8,
        lc_speed_gain=4.0,
        model="LC2013",
        lane_change_mode="strategic",  # TODO: check-is there a better way to change lanes?
        lc_keep_right=0.8
    )

    pedestrian_params = None
    if pedestrians:
        pedestrian_params = PedestrianParams()
        for i in range(20):
            pedestrian_params.add(
                ped_id='ped_{}'.format(i),
                depart_time='0.00',
                start='(1.2)--(1.1)',
                end='(1.1)--(1.0)',
                depart_pos=str(43 + 0.5 * i))

    vehicles = VehicleParams()

    vehicles.add(
        veh_id="av",
        routing_controller=(GridRouter, {}),
        car_following_params=SumoCarFollowingParams(
            min_gap=2.5,
            decel=7.5,  # avoid collisions at emergency stops
            speed_mode="aggressive",
        ),
        acceleration_controller=(RuleBasedIntersectionController, {}),
        lane_change_params=lane_change_params,
        num_vehicles=4)

    # For now, just have the one human car and one pedestrian

    # vehicles.add(
    #     veh_id="obstacle",
    #     routing_controller=(GridRouter, {}),
    #     car_following_params=SumoCarFollowingParams(
    #         min_gap=2.5,
    #         decel=7.5,  # avoid collisions at emergency stops
    #         speed_mode="right_of_way",
    #         max_speed=0.000001
    #     ),
    #     lane_change_params=lane_change_params,
    #     num_vehicles=num_cars_top)

    # vehicles.add(
    #     veh_id="rl",
    #     routing_controller=(GridRouter, {}),
    #     car_following_params=SumoCarFollowingParams(
    #         min_gap=2.5,
    #         decel=7.5,  # avoid collisions at emergency stops
    #         speed_mode="right_of_way",
    #     ),
    #     lane_change_params=lane_change_params,
    #     num_vehicles=num_cars_right)

    env_params = EnvParams(additional_params=ADDITIONAL_ENV_PARAMS)

    additional_net_params = {
        "grid_array": grid_array,
        "speed_limit": 35,
        "horizontal_lanes": 1,
        "vertical_lanes": 1,
        "randomize_routes": True
    }

    initial_config, net_params = get_non_flow_params(
        enter_speed=v_enter,
        add_net_params=additional_net_params,
        pedestrians=pedestrians)

    network = Bayesian0Network(
        name="bayesian_0",
        vehicles=vehicles,
        net_params=net_params,
        pedestrians=pedestrian_params,
        initial_config=initial_config)

    env = Bayesian0NoGridEnv(env_params, sim_params, network)
    return Experiment(env)


if __name__ == "__main__":
    # check for pedestrians
    parser = argparse.ArgumentParser()
    parser.add_argument("--pedestrians",
                        help="use pedestrians, sidewalks, and crossings in the simulation",
                        action="store_true")
    # wonder if it's better to call the argument the actual experiment file ... I'll be using Bayesian0Experiement.py
    parser.add_argument("--collect_data",
                        help="collect training data from this experiment by using bayesian 0 experiment rather than Experiment",
                        action="store_true")

    parser.add_argument("--render",
                        help="render the SUMO simulation",
                        action="store_true")

    args = parser.parse_args()
    pedestrians = args.pedestrians
    collect_data = args.collect_data
    render = args.render
    # import the experiment variable
    exp = bayesian_0_example(render=render, pedestrians=pedestrians, collect_data=collect_data)
    # run for a set number of rollouts / time steps
    exp.run(40, 600, multiagent=True)
