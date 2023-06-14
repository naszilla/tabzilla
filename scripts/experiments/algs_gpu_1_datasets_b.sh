#!/bin/bash

# load functions
source ../utils.sh

##############################
# begin: EXPERIMENT PARAMETERS

# this defines MODELS_ENVS
source ../ALGS_GPU_1.sh

# this defines DATASETS
source ../DATASETS_B.sh

name=algs-gpu-1-datasets-b

# base name for the gcloud instances
instance_base=$name

# experiment name (will be appended to results files)
experiment_name=$name

# maximum number of experiments (background processes) that can be running
MAX_PROCESSES=10

# config file
config_file=/home/shared/tabzilla/TabZilla/tabzilla_experiment_config_gpu.yml

# results file: check for results here before launching each experiment
result_log=/home/shared/tabzilla/TabZilla/result_log.txt

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

    # if the experiment is already in the result log, skip it
    if grep -Fxq "${DATASETS[j]},${model},${experiment_name}" ${result_log}; then
      echo "experiment found in logs. skipping. dataset=${DATASETS[j]}, model=${model}, expt=${experiment_name}"
      continue
    fi

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

    run_experiment_gpu "${model}" ${DATASETS[j]} ${env} ${instance_base}-${i}-${j} ${experiment_name} ${config_file} >> ${LOG_DIR}/log_${i}_${j}_$(date +"%m%d%y_%H%M%S").txt 2>&1 &
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