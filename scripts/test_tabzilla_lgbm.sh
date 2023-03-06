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
  "LightGBM:$GBDT_ENV"
  )

CONFIG_FILE=tabzilla_experiment_config.yml



DATASETS=(
 openml__Fashion-MNIST__146825
 openml__albert__189356
 openml__christine__168908
 openml__dilbert__168909
 openml__one-hundred-plants-texture__9956
 openml__poker-hand__9890
 openml__robert__168332
 openml__CIFAR_10__167124
 openml__Devnagari-Script__167121
 openml__guillermo__168337
 openml__helena__168329
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
