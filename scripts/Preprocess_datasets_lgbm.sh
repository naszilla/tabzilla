#!/bin/bash

# small test script to make sure that TabSurvey code and environments work.

cd /home/shared/tabzilla/TabSurvey

#ENV_NAME=torch

# name of the model/algorithm
#MODEL_NAME=DeepFM

# all datasets should be in this folder. the dataset folder should be in ${DATASET_BASE_DIR}/<dataset-name>
DATASET_BASE_DIR=./datasets

##########################################################
# define lists of datasets and models to evaluate them on

# MODELS_ENVS=(
#   "LinearModel:$SKLEARN_ENV"
#   "KNN:$SKLEARN_ENV"
#   "DecisionTree:$SKLEARN_ENV"
#   )

#CONFIG_FILE=tabzilla_experiment_config.yml


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

for DATASET_NAME in ${DATASETS[@]}; do
    printf "\n|----------------------------------------------------------------------------\n"
    printf "| starting dataset ${DATASET_NAME}\n"

    #conda activate ${ENV_NAME}

    #DATSET_DIR=${DATASET_BASE_DIR}/${DATASET_NAME}
    python tabzilla_data_preprocessing.py --dataset_name ${DATASET_NAME}

    # zip results into a new directory, and remove unzipped results
    #zip -r results_${dataset}_${model}.zip ./results
    #rm -r ./results

    #conda deactivate
done