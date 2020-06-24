"""Transfer tests using the prediction environment"""
import argparse
from datetime import datetime
import gym
import numpy as np
import os
import sys
import time

from flow.core.util import emission_to_csv

def run_env(args, env, name):

    rets = []

    sim_params = env.sim_params
    # pick your rendering mode
    if args.render_mode == 'sumo_web3d':
        sim_params.num_clients = 2
        sim_params.render = False
    elif args.render_mode == 'drgb':
        sim_params.render = 'drgb'
        sim_params.pxpm = 4
    elif args.render_mode == 'sumo_gui':
        sim_params.render = True
        print('NOTE: With render mode {}, an extra instance of the SUMO GUI '
              'will display before the GUI for visualizing the result. Click '
              'the green Play arrow to continue.'.format(args.render_mode))
    elif args.render_mode == 'no_render':
        sim_params.render = False
    if args.save_render:
        sim_params.render = 'drgb'
        sim_params.pxpm = 4
        sim_params.save_render = True
    env.sim_params = sim_params

    env.restart_simulation(
        sim_params=sim_params, render=sim_params.render)

    # Simulate and collect metrics
    final_outflows = []
    final_inflows = []
    mean_speed = []
    std_speed = []
    num_pedestrian_crash = 0
    completion = 0
    for i in range(args.num_rollouts):
        collision = False
        vel = []
        state = env.reset()
        ret = 0
        for _ in range(env.env_params.horizon):
            vehicles = env.unwrapped.k.vehicle
            pedestrian = env.unwrapped.k.pedestrian
            vel.append(np.mean(vehicles.get_speed(vehicles.get_ids())))
            state, reward, done, _ = env.step(None)
            ret += reward

            for rl_id in vehicles.get_rl_ids():
                num_collision = len(vehicles.get_pedestrian_crash(rl_id, pedestrian))
                num_pedestrian_crash += num_collision
                if num_collision > 0:
                    collision = True

            if done or 'av_0' not in vehicles.get_ids() or collision:
                # we made it to the end before the rollout terminated and we didn't collide
                if env.time_counter * env.sim_step < 40.0 and not collision:
                    completion += 1
                break

        rets.append(ret)
        outflow = vehicles.get_outflow_rate(500)
        final_outflows.append(outflow)
        inflow = vehicles.get_inflow_rate(500)
        final_inflows.append(inflow)
        mean_speed.append(np.mean(vel))
        std_speed.append(np.std(vel))
        print('Round {}, Return: {}'.format(i, ret))

    print('==== Summary of results ====')
    print("Return:")
    print(mean_speed)
    print(rets)
    print('Average, std: {}, {}'.format(
        np.mean(rets), np.std(rets)))

    print("\nSpeed, mean (m/s):")
    print(mean_speed)
    print('Average, std: {}, {}'.format(np.mean(mean_speed), np.std(
        mean_speed)))
    print("\nSpeed, std (m/s):")
    print(std_speed)
    print('Average, std: {}, {}'.format(np.mean(std_speed), np.std(
        std_speed)))

    # Compute arrival rate of vehicles in the last 500 sec of the run
    print("\nOutflows (veh/hr):")
    print(final_outflows)
    print('Average, std: {}, {}'.format(np.mean(final_outflows),
                                        np.std(final_outflows)))
    # Compute departure rate of vehicles in the last 500 sec of the run
    print("Inflows (veh/hr):")
    print(final_inflows)
    print('Average, std: {}, {}'.format(np.mean(final_inflows),
                                        np.std(final_inflows)))

    print("Number of pedestrian crashes:")
    print(num_pedestrian_crash)

    print("Completion fraction: {}/{}".format(completion, args.num_rollouts))

    file_path = os.path.expanduser('~/generalization')
    # write the results to a folder for keeping track
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    print("WRITING RESULTS TO {}".format(file_path))
    with open(os.path.join(file_path, name), 'w') as file:
        file.write("Num_trials, successes, collisions\n")
        file.write("{} {} {}".format(args.num_rollouts, completion, num_pedestrian_crash))
        file.close()

    # terminate the environment
    env.unwrapped.terminate()

    # if prompted, convert the emission file into a csv file
    if args.gen_emission:
        time.sleep(0.1)

        dir_path = os.path.dirname(os.path.realpath(__file__))
        emission_filename = '{0}-emission.xml'.format(env.network.name)

        emission_path = \
            '{0}/test_time_rollout/{1}'.format(dir_path, emission_filename)

        # convert the emission file into a csv file
        emission_to_csv(emission_path)

        # print the location of the emission csv file
        emission_path_csv = emission_path[:-4] + ".csv"
        print("\nGenerated emission file at " + emission_path_csv)

        # delete the .xml version of the emission file
        os.remove(emission_path)

def run_transfer(args):
    # run transfer on the bayesian 1 env first
    from examples.rllib.multiagent_exps.exp_configs.prediction_configs.bayesian_1_config import make_env as make_1_env
    env = make_1_env()
    if args.num_rollouts > 1:
        env.sim_params.restart_instance = True
    run_env(args, env, name="bayesian_1_test_mpc")

    # run transfer on the bayesian 3 env
    # from examples.rllib.multiagent_exps.exp_configs.prediction_configs.bayesian_3_config import make_env as make_3_env
    # env = make_3_env()
    # if args.num_rollouts > 1:
    #     env.sim_params.restart_instance = True
    # run_env(args, env, name="bayesian_3_test_mpc")

    # from examples.rllib.multiagent_exps.exp_configs.prediction_configs.bayesian_4_config import make_env as make_4_env
    # env = make_4_env()
    # if args.num_rollouts > 1:
    #     env.sim_params.restart_instance = True
    # run_env(args, env, name="bayesian_4_test_mpc")


def create_parser():
    """Create the parser to capture CLI arguments."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='[Flow] Evaluates the MPC agents')

    # optional input parameters
    parser.add_argument(
        '--num_rollouts',
        type=int,
        default=1,
        help='The number of rollouts to visualize.')
    parser.add_argument(
        '--gen_emission',
        action='store_true',
        help='Specifies whether to generate an emission file from the '
             'simulation')
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='Specifies whether to use the \'evaluate\' reward '
             'for the environment.')
    parser.add_argument(
        '--render_mode',
        type=str,
        default='sumo_gui',
        help='Pick the render mode. Options include sumo_web3d, '
             'rgbd and sumo_gui')
    parser.add_argument(
        '--save_render',
        action='store_true',
        help='Saves a rendered video to a file. NOTE: Overrides render_mode '
             'with pyglet rendering.')
    parser.add_argument(
        '--horizon',
        type=int,
        help='Specifies the horizon.')
    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    run_transfer(args)
