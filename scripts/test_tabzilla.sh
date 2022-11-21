#!/bin/bash

# small test script to make sure that TabSurvey code and environments work.

cd /home/ramyasri/tabzilla/TabSurvey

SKLEARN_ENV="sklearn"
GBDT_ENV="gbdt"
TORCH_ENV="torch"
KERAS_ENV="tensorflow"

# all datasets should be in this folder. the dataset folder should be in ${DATASET_BASE_DIR}/<dataset-name>
DATASET_BASE_DIR=./datasets

##########################################################
# define lists of datasets and models to evaluate them on

MODELS_ENVS=(
  "SVM:$SKLEARN_ENV"
  )

CONFIG_FILE=tabzilla_experiment_config.yml

DATASETS=()
search_dir=/home/ramyasri/tabzilla/TabSurvey/datasets
for entry in `ls $search_dir`
do
  DATASETS+=($entry)
done

# DATASETS=(
#   openml__california__361089

# )

conda init bash
eval "$(conda shell.bash hook)"

for dataset in "${DATASETS[@]}"; do
  printf "\n|----------------------------------------------------------------------------\n"
  printf "| starting dataset ${dataset}\n"
  for model_env in "${MODELS_ENVS[@]}"; do

      model="${model_env%%:*}"
      env="${model_env##*:}"

      printf "\n||----------------------------------------------------------------------------\n"
      printf '|| Training %s on dataset %s in env %s\n\n' "$model" "$dataset" "$env"

      conda activate ${env}

      dataset_dir=${DATASET_BASE_DIR}/${dataset}
      python tabzilla_experiment.py --experiment_config ${CONFIG_FILE} --dataset_dir ${dataset_dir} --model_name ${model}

      # zip results into a new directory, and remove unzipped results
      zip -r results_${dataset}_${model}.zip ./results
      rm -r ./results

      conda deactivate

  done
done
