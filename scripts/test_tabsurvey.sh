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

# MODELS=(
#   "LinearModel:$SKLEARN_ENV"
#   "KNN:$SKLEARN_ENV"
#   "DecisionTree:$SKLEARN_ENV"
#   )

CONFIG_FILES=(
  tabzilla_experiment_config.yml
  tabzilla_experiment_config_1.yml
  tabzilla_experiment_config_2.yml
)


DATASETS=(
  openml__california__361089
)

# conda init bash
eval "$(conda shell.bash hook)"

# run only with sklearn
env=sklearn

for config in "${CONFIG_FILES[@]}"; do

    printf "\n\n|----------------------------------------------------------------------------\n"
    printf "| Running experiment with config file ${config} with env '${env}'"

    conda activate ${env}

    python tabzilla_experiment.py --experiment_config ${config}

    conda deactivate

  done

done
