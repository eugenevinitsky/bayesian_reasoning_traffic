i=0
for run_script in rllib/ppo_runner.py; do
    i=$((i+1))
    echo $i
    echo ${run_script}
done
