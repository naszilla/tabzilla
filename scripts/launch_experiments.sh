#!/bin/bash

# load functions
source utils.sh

###################
# set trap
# this will sync the log files, and delete all instances

mkdir ${PWD}/logs
LOG_DIR=${PWD}/logs
trap "sync_logs ${LOG_DIR}; delete_instances" EXIT
INSTANCE_LIST=()

###################
# define parameters

SKLEARN_ENV="sklearn"
GBDT_ENV="gbdt"
TORCH_ENV="torch"
KERAS_ENV="tensorflow"

MODELS_ENVS=(
  "LinearModel:$SKLEARN_ENV"
  "KNN:$SKLEARN_ENV"
  "DecisionTree:$SKLEARN_ENV"
  )

DATASETS=(
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
    echo "MODEL_ENV: ${model_env}"
    echo "MODEL: ${model}"
    echo "ENV: ${env}"
    echo "DATASET: ${DATASETS[j]}"

    echo "run_experiment "${model}" ${DATASETS[j]} ${env} ${instance_base}-${i}-${j} >> ${LOG_DIR}/log_${i}_${j}_$(date +"%m%d%y_%H%M%S").txt 2>&1 &
    num_experiments=$((num_experiments + 1))"

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