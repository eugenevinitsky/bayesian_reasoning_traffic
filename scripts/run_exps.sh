#!/bin/bash

ray exec ray_autoscale.yaml "python bayesian_reasoning_traffic/examples/rllib/multiagent_exps/bayesian_0_no_grid_env.py \
--use_s3 --grid_search --n_cpus 1" --start --stop --cluster-name=ev_bay_1