"""There is a vehicle obscuring a pedestrian that conflicts with your path."""

from flow.envs.ring.accel import AccelEnv, AccelWithQueryEnv, ADDITIONAL_ENV_PARAMS
from flow.controllers.bayesian_predict_controller import BayesianPredictController
from flow.networks import Bayesian1Network
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.params import SumoCarFollowingParams, VehicleParams
from flow.core.params import PedestrianParams
from flow.core.experiment import Experiment

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


def make_env():
    """
    Generate the flow params for the experiment.
    Parameters
    ----------
    Returns
    -------
    env
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
        veh_id="av",
        acceleration_controller=(BayesianPredictController, {}),
        routing_controller=(GridRouter, {}),
        car_following_params=SumoCarFollowingParams(
            min_gap=2.5,
            decel=7.5,  # avoid collisions at emergency stops
            speed_mode="aggressive",
        ),
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
                # "randomize_routes": True,
            }
    )
    net = Bayesian1Network(
        name="bayesian_1",
        vehicles=vehicles,
        net_params=net_params,
        pedestrians=pedestrian_params,
        initial_config=initial_config)

    # sumo-related parameters (see flow.core.params.SumoParams)
    sim = SumoParams(
        restart_instance=False,
        sim_step=1.0,
        render=True,
    )

    env_params = EnvParams(
        horizon=500,
        # environment related parameters (see flow.core.params.EnvParams)
        additional_params=ADDITIONAL_ENV_PARAMS,
    )

    env = AccelWithQueryEnv(env_params, sim, net)
    env.query_env = AccelEnv(env_params, sim, net)
    return env

if __name__ == '__main__':
    env = make_env()
    exp = Experiment(env)
    exp.run(1, 1500)