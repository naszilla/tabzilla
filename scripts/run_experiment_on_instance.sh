#! /bin/bash
set -e

###############################################################################################################
# This script runs a single experiment on a gcloud instance.
# This instance needs to have a (not-necessarily-updated) TabZilla codebase in /home/shared/tabzilla
#
# NOTE: this script requires that two environment variables are defined
# (use `export <var>=value` to define these): DATASET_DIR, and MODEL_NAME. see below:
#
#  - ENV_NAME <- the conda env that should be used (this should go with the model)
#  - MODEL_NAME <- name of the ML model to use
#  - DATASET_NAME <- the name of the dataset to use for the experiment
#  - EXPERIMENT_NAME <- name of the experiment. this will be appended to the result file name
#  - CONFIG_FILE <- name of the config file (yml)
# 
###############################################################################################################

#############################################
# make sure environment variables are defined

if [ -n "$ENV_NAME" ]; then
  echo "ENV_NAME: $ENV_NAME"
else
  echo "ENV_NAME string not defined" 1>&2
fi

if [ -n "$MODEL_NAME" ]; then
  echo "MODEL_NAME: $MODEL_NAME"
else
  echo "MODEL_NAME not defined" 1>&2
fi

if [ -n "$DATASET_NAME" ]; then
  echo "DATASET_NAME: $DATASET_NAME"
else
  echo "DATASET_NAME string not defined" 1>&2
fi

if [ -n "$EXPERIMENT_NAME" ]; then
  echo "EXPERIMENT_NAME: $EXPERIMENT_NAME"
else
  echo "EXPERIMENT_NAME string not defined" 1>&2
fi

if [ -n "$CONFIG_FILE" ]; then
  echo "CONFIG_FILE: $CONFIG_FILE"
else
  echo "CONFIG_FILE string not defined" 1>&2
fi

###############
# prepare conda

source /opt/conda/bin/activate
conda init


################
# run experiment

printf 'running experiment with model %s on dataset %s in env %s\n\n' "$MODEL_NAME" "$DATASET_NAME" "$ENV_NAME"

# use the env specified in ENV_NAME
conda activate ${ENV_NAME}

# all datasets should be in this folder. the dataset folder should be in ${DATASET_BASE_DIR}/<dataset-name>
DATASET_BASE_DIR=./datasets
DATSET_DIR=${DATASET_BASE_DIR}/${DATASET_NAME}

# run the experiment using command line args
cd /home/shared/tabzilla/TabZilla

python tabzilla_experiment.py --experiment_config ${CONFIG_FILE} --dataset_dir ${DATSET_DIR} --model_name ${MODEL_NAME}

# zip results
zip -jr results.zip ./results

# add a timestamp and a random string to the end of the filename, to avoid collisions
result_file=${EXPERIMENT_NAME}_$(date +"%m%d%y_%H%M%S")_$(openssl rand -hex 2).zip
mv ./results.zip ./${result_file}


###############################
# save results to gcloud bucket

gsutil cp ./${result_file} gs://tabzilla-results/results/${DATASET_NAME}/${MODEL_NAME}/${result_file}
