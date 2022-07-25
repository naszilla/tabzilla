#!/bin/bash

# small test script to make sure that TabSurvey code and environments work.
# this script should be run from directory tabzilla
cd ./TabSurvey

TABZILLA_DIR=/home/duncan/tabzilla
SEARCH_CONFIG=${TABZILLA_DIR}/TabSurvey/tabzilla_search_config.yml

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

DATASET_NAMES=( 
  CaliforniaHousing
  Covertype
)

# conda init bash
eval "$(conda shell.bash hook)"

for dataset_name in "${DATASET_NAMES[@]}"; do

  for model in "${!MODELS[@]}"; do
    printf "\n\n----------------------------------------------------------------------------\n"
    printf 'Training %s with %s in env %s\n\n' "$model" "$dataset_name" "${MODELS[$model]}"

    # pre-process dataset. overwrite any existing dataset using flag --overwrite
    conda activate base
    python tabzilla_data_preprocessing.py --dataset_name ${dataset_name} --overwrite

    # the dataset will be located here
    dataset_dir=${TABZILLA_DIR}/TabSurvey/datasets/${dataset_name}

    conda activate "${MODELS[$model]}"

    python tabzilla_experiment.py --dataset_dir  ${dataset_dir} --search_config ${SEARCH_CONFIG} --model_name "${model}"
    

    conda deactivate

  done

done

