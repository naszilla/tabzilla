#! /bin/bash

#Ensures that the script will fail if any command fails
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
cd /scratch/bf996/tabzilla/TabZilla

#############################################
# make sure environment variables are defined

# define the conda env that should be used {sklearn | gbdt | torch | tensorflow}
ENV_NAME=torch

# name of the model/algorithm
MODEL_NAME=TabPFNModel

# name of the dataset
DATASET_NAME=/openml__anneal__2867

#TOO MANY FEATURES
#/openml__mfeat-pixel__146824

#TOO MANY ROWS
#DATASET_NAME=/openml__albert__189356

# name of the config file to use
#CONFIG_FILE=tabzilla_experiment_config.yml
CONFIG_FILE=tabzilla_experiment_config_gpu.yml

# all datasets should be in this folder. the dataset folder should be in ${DATASET_BASE_DIR}/<dataset-name>
DATASET_BASE_DIR=./datasets
DATSET_DIR=${DATASET_BASE_DIR}/${DATASET_NAME}

#log directory for console outputs
LOG_DIR=/scratch/bf996/tabzilla/TabZilla/logs
RESULTS_DIR=./results
SAVE_FILE=$(date +"%m%d%y_%H%M%S")
#########################################################

singularity exec --nv --overlay /scratch/bf996/singularity_containers/tabzilla.ext3:ro /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif   /bin/bash -c "\
source /ext3/env.sh; \
printf 'running experiment with model %s on dataset %s in env %s\n\n' "$MODEL_NAME" "$DATASET_NAME" "$ENV_NAME"; \
conda activate ${ENV_NAME}; \
python tabzilla_experiment.py --experiment_config ${CONFIG_FILE} --dataset_dir ${DATSET_DIR} --model_name ${MODEL_NAME} \
>> ${LOG_DIR}/log_${SAVE_FILE}.txt; \
zip -jr results.zip ${RESULTS_DIR}; \
RESULT_FILE=${SAVE_FILE}_$(openssl rand -hex 2).zip; \
printf 'saving results to %s\n\n' "$RESULT_FILE"; \
mv ./results.zip ./${RESULT_FILE}; \
rm -rf RESULTS_DIR;"