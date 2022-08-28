#!/bin/bash

# load functions
source utils.sh

###################
# set trap
# this will sync the log files, and delete all instances

mkdir ${PWD}/logs
LOG_DIR=${PWD}/logs
trap "sync_logs ${LOG_DIR}; delete_instances instance_list" EXIT
instance_list=()

###################
# define parameters

alg_list=(
    KNN
    DecisionTree
)

dataset_list=(
    openml__california__361089
    openml__MiceProtein_146800
)

# base name for the gcloud instances
instance_base=tztest

# maximum number of experiments (background processes) that can be running
MAX_PROCESSES=10

#################
# run experiments

num_experiments=0
for i in ${!alg_list[@]};
do
  for j in ${!dataset_list[@]};
  do

    instance_name=${instance_base}-${i}-${j}

    run_experiment "${arg_str}" ${split_path_on_bucket} ${instance_base}-${i}-${j} >> ${LOG_DIR}/log_${i}_${j}_$(date +"%m%d%y_%H%M%S").txt 2>&1 &
    num_experiments=$((num_experiments + 1))

    # add instance name to the instance list
    instance_list+=("${instance_name}")

    echo "launched instance ${instance_base}-${i}-${j}. (job number ${num_experiments})"
    sleep 1

    # if we have started MAX_PROCESSES experiments, wait for them to finish
    wait_until_processes_finish $MAX_PROCESSES

  done
done

echo "still waiting for processes to finish..."
wait
echo "done."