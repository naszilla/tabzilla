#!/bin/bash

# small test script to make sure that TabSurvey code and environments work.

cd ./TabSurvey

SKLEARN_ENV="sklearn"
GBDT_ENV="gbdt"
TORCH_ENV="torch"
KERAS_ENV="tensorflow"

# all datasets should be in this folder. the dataset folder should be in ${DATASET_BASE_DIR}/<dataset-name>
DATASET_BASE_DIR=./datasets

##########################################################
# define lists of datasets and models to evaluate them on

MODELS_ENVS=(
  "CatBoost:$GBDT_ENV"
  )

CONFIG_FILE=tabzilla_experiment_config.yml



DATASETS=(
 openml__isolet__3481
 openml__pendigits__32
 openml__robert__168332
 openml__solar-flare__2068
 openml__CIFAR_10__167124
 openml__Devnagari-Script__167121
 openml__covertype__7593
 openml__guillermo__168337
 openml__helena__168329
 openml__riccardo__168338
 openml__shuttle__146212
)

# conda init bash
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
