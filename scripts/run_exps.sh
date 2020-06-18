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