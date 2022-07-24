#!/bin/bash

# small test script to make sure that TabSurvey code and environments work.
# this script should be run from directory tabzilla
cd ./TabSurvey

CONFIG_DIR=/home/duncan/tabzilla/TabSurvey/temp
SEARCH_CONFIG=${CONFIG_DIR}/general.yml

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

DATASET_DIRS=( "${CONFIG_DIR}/cal_housing"
          )

# conda init bash
eval "$(conda shell.bash hook)"

for dataset_dir in "${DATASET_DIRS[@]}"; do

  for model in "${!MODELS[@]}"; do
    printf "\n\n----------------------------------------------------------------------------\n"
    printf 'Training %s with %s in env %s\n\n' "$model" "$dataset_config" "${MODELS[$model]}"

    conda activate "${MODELS[$model]}"

    python tabzilla_experiment.py --dataset_dir  ${dataset_dir} --search_config ${SEARCH_CONFIG} --model_name "${model}"
    

    conda deactivate

  done

done

