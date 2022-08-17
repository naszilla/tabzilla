#!/bin/bash

# small test script to make sure that TabSurvey code and environments work.

cd ./TabSurvey

SKLEARN_ENV="sklearn"
GBDT_ENV="gbdt"
TORCH_ENV="torch"
KERAS_ENV="tensorflow"

# search parameters
SEARCH_CONFIG=./tabzilla_search_config.yml


##########################################################
# define lists of datasets and models to evaluate them on

MODELS=(
  "LinearModel:$SKLEARN_ENV"
  "KNN:$SKLEARN_ENV"
  "DecisionTree:$SKLEARN_ENV"
  )


DATASETS=(
  openml__california__361089
)

# conda init bash
eval "$(conda shell.bash hook)"

for dataset in "${DATASETS[@]}"; do
    printf "\n----------------------------------------------------------------------------\n"
    printf "pre-processing dataset: ${dataset}...\n"

    conda activate openml

    # pre-process dataset
    python tabzilla_data_preprocessing.py --dataset_name ${dataset}

    # this will write the dataset to directory TabSurvey/datasets/${dataset}
    DATASET_DIR=./datasets/${dataset}

    conda deactivate

  for model_env in "${MODELS[@]}"; do

    model="${model_env%%:*}"
    env="${model_env##*:}"

    printf "\n\n|----------------------------------------------------------------------------\n"
    # printf 'Training %s on dataset %s in env %s\n\n' "$model" "$dataset" "${MODELS[$model]}"
    printf '| Training %s on dataset %s in env %s\n\n' "$model" "$dataset" "$env"

    conda activate "${env}"

    python tabzilla_experiment.py --dataset_dir ${DATASET_DIR} --search_config ${SEARCH_CONFIG} --model_name "$model"

    conda deactivate

  done

done
