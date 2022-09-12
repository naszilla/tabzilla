#!/bin/bash

# load functions
source utils.sh

##############################
# begin: EXPERIMENT PARAMETERS

MODELS_ENVS=(
  "LinearModel:$SKLEARN_ENV"
  "KNN:$SKLEARN_ENV"
  "DecisionTree:$SKLEARN_ENV"
  )

DATASETS=(
    openml__california__361089
    openml__MiceProtein__146800
)

# base name for the gcloud instances
instance_base=tztest

# experiment name (will be appended to results files)
experiment_name=test-experiment

# maximum number of experiments (background processes) that can be running
MAX_PROCESSES=10

# end: EXPERIMENT PARAMETERS
############################

####################
# begin: bookkeeping

# make a log directory
mkdir ${PWD}/logs
LOG_DIR=${PWD}/logs

###################
# set trap
# this will sync the log files, and delete all instances

trap "sync_logs ${LOG_DIR} ${experiment_name}; delete_instances" EXIT
INSTANCE_LIST=()

# end: bookkeeping
##################


#################
# run experiments

num_experiments=0
for i in ${!MODELS_ENVS[@]};
do
  for j in ${!DATASETS[@]};
  do
    model_env="${MODELS_ENVS[i]}"
    model="${model_env%%:*}"
    env="${model_env##*:}"

    instance_name=${instance_base}-${i}-${j}

    # args:
    # $1 = model name
    # $2 = dataset name
    # $3 = env name
    # $4 = instance name
    # $5 = experiment name
    echo "MODEL_ENV: ${model_env}"
    echo "MODEL: ${model}"
    echo "ENV: ${env}"
    echo "DATASET: ${DATASETS[j]}"
    echo "EXPERIMENT_NAME: ${experiment_name}"

    run_experiment "${model}" ${DATASETS[j]} ${env} ${instance_base}-${i}-${j} ${experiment_name} >> ${LOG_DIR}/log_${i}_${j}_$(date +"%m%d%y_%H%M%S").txt 2>&1 &
    num_experiments=$((num_experiments + 1))

    # add instance name to the instance list
    INSTANCE_LIST+=("${instance_name}")

    echo "launched instance ${instance_base}-${i}-${j}. (job number ${num_experiments})"
    sleep 1

    # if we have started MAX_PROCESSES experiments, wait for them to finish
    wait_until_processes_finish $MAX_PROCESSES

  done
done

echo "still waiting for processes to finish..."
wait
echo "done."