#! /bin/bash
set -e

###############################################################################################################
# This script runs a single experiment on a gcloud instance, on an ad-hoc basis.

# This instance needs to have a (not-necessarily-updated) TabZilla codebase in /home/shared/tabzilla
#
# NOTE: this script requires that two environment variables are defined
# (use `export <var>=value` to define these): DATASET_DIR, and MODEL_NAME. see below:
#
#  - ENV_NAME <- the conda env that should be used (this should go with the model)
#  - MODEL_NAME <- name of the ML model to use
#  - DATASET_NAME <- the name of the dataset to use for the experiment
#
###############################################################################################################

# run from this directory
cd /home/shared/tabzilla/TabZilla

#############################################
# make sure environment variables are defined

# define the conda env that should be used {sklearn | gbdt | torch | tensorflow}
ENV_NAME=gbdt

# name of the model/algorithm
MODEL_NAME=XGBoost

# name of the dataset
DATASET_NAME=/openml__arrhythmia__5

#########################################################
# prepare conda, in case it has not already been prepared

source /opt/conda/bin/activate
conda init

################
# run experiment

printf 'running experiment with model %s on dataset %s in env %s\n\n' "$MODEL_NAME" "$DATASET_NAME" "$ENV_NAME"

# use the env specified in ENV_NAME
conda activate ${ENV_NAME}

# search parameters - this is the default
# CONFIG_FILE=tabzilla_experiment_config.yml
CONFIG_FILE=tabzilla_experiment_config_gpu.yml

# all datasets should be in this folder. the dataset folder should be in ${DATASET_BASE_DIR}/<dataset-name>
DATASET_BASE_DIR=./datasets
DATSET_DIR=${DATASET_BASE_DIR}/${DATASET_NAME}

python tabzilla_experiment.py --experiment_config ${CONFIG_FILE} --dataset_dir ${DATSET_DIR} --model_name ${MODEL_NAME}

# results will be written to /home/shared/tabzilla/TabZilla/results
# you can zip them if you'd like:
# > zip -r results.zip ./results
