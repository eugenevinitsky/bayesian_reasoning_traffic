#!/bin/bash

##ray exec ray_autoscale.yaml "python bayesian_reasoning_traffic/examples/rllib/multiagent_exps/bayesian_0_no_grid_env.py \
##--use_s3 --grid_search --n_cpus 1 --run_transfer_tests" --start --stop --cluster-name=ev_bay_1 --tmux
#
#ray exec ray_autoscale.yaml "python bayesian_reasoning_traffic/examples/rllib/multiagent_exps/bayesian_0_no_grid_env.py \
#--use_s3 --grid_search --n_cpus 1 --run_transfer_tests --use_lstm --exp_title lstm_bseline" --start --stop --cluster-name=ev_bay_2 --tmux

#########################################################################################################################
# 5/15 exps
ray exec ray_autoscale.yaml "python bayesian_reasoning_traffic/examples/rllib/multiagent_exps/bayesian_0_no_grid_env.py \
--use_s3 --grid_search --n_cpus 1 --run_transfer_tests" --start --stop --cluster-name=ev_bay_1 --tmux

ray exec ray_autoscale.yaml "python bayesian_reasoning_traffic/examples/rllib/multiagent_exps/bayesian_0_no_grid_env.py \
--use_s3 --grid_search --n_cpus 1 --run_transfer_tests --use_lstm --exp_title lstm_bseline" --start --stop --cluster-name=ev_bay_2 --tmux


####################################################################################################
# 8/31 tests of fully homogeneous policy
####################################################################################################
ray exec ray_autoscale.yaml "python bayesian_reasoning_traffic/examples/rllib/multiagent_exps/bayesian_0_no_grid_env.py \
--checkpoint_freq 100 --n_iterations 400 --algo PPO --n_rollouts 240 --n_cpus 10 --exp_title \
l0_training_ppo_veh4_rollout240_30pen_nofilter_ped_onlyrl --use_s3 --grid_search --ped_collision_penalty -30 \
--num_vehicles 4 --run_mode test --pedestrians --only_rl" --start --stop \
--cluster-name=ev_bayes_24 --tmux