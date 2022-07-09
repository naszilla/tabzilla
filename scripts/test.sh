#!/bin/bash

# small test script to make sure that TabSurvey code and environments work.
# this script should be run from directory tabzilla
cd ./TabSurvey

CONFIG_DIR=/home/duncan/tabzilla/tabzilla_config_library
GENERAL_CONFIG=${CONFIG_DIR}/general.yml

N_TRIALS=2
EPOCHS=3

SKLEARN_ENV="sklearn"
GBDT_ENV="gbdt"
TORCH_ENV="torch"
KERAS_ENV="tensorflow"

declare -A MODELS
MODELS=( 
  # ["LinearModel"]=$SKLEARN_ENV
         ["KNN"]=$SKLEARN_ENV
        #  ["DecisionTree"]=$SKLEARN_ENV
          )

DATASET_CONFIGS=( "${CONFIG_DIR}/datasets/adult.yml"
          "${CONFIG_DIR}/datasets/california_housing.yml"
          )

# conda init bash
eval "$(conda shell.bash hook)"

for dataset_config in "${DATASET_CONFIGS[@]}"; do

  for model in "${!MODELS[@]}"; do
    printf "\n\n----------------------------------------------------------------------------\n"
    printf 'Training %s with %s in env %s\n\n' "$model" "$dataset_config" "${MODELS[$model]}"

    conda activate "${MODELS[$model]}"

    # python tabzilla_experiment.py --config "$config" --model_name "$model" --n_trials $N_TRIALS --epochs $EPOCHS --optimize_hyperparameters
    python tabzilla_experiment.py --dataset_config  ${dataset_config} --general_config ${GENERAL_CONFIG} --model_name "${model}"
    

    conda deactivate

  done

done

