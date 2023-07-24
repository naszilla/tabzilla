#! /bin/bash

#Ensures that the script will fail if any command fails
set -e

# run from this directory
cd /scratch/bf996/tabzilla/TabZilla

#############################################
# make sure environment variables are defined

# define the conda env that should be used {sklearn | gbdt | torch | tensorflow}
ENV_NAME=gbdt

# name of the model/algorithm
MODEL_NAME=XGBoost

# name of the dataset
DATASET_NAME=/openml__APSFailure__168868

# name of the config file to use
#CONFIG_FILE=tabzilla_experiment_config.yml
CONFIG_FILE=tabzilla_experiment_config_gpu.yml

# all datasets should be in this folder. the dataset folder should be in ${DATASET_BASE_DIR}/<dataset-name>
DATASET_BASE_DIR=./datasets
DATSET_DIR=${DATASET_BASE_DIR}/${DATASET_NAME}

#########################################################
# prepare conda, in case it has not already been prepared

singularity exec --nv --overlay /scratch/bf996/singularity_containers/tabzilla.ext3:ro /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif   /bin/bash -c "\
source /ext3/env.sh; \
printf 'running experiment with model %s on dataset %s in env %s\n\n' "$MODEL_NAME" "$DATASET_NAME" "$ENV_NAME"; \
conda activate ${ENV_NAME}; \
python tabzilla_experiment.py --experiment_config ${CONFIG_FILE} --dataset_dir ${DATSET_DIR} --model_name ${MODEL_NAME}"
