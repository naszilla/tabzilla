#!/bin/bash

# small test script to make sure that TabSurvey code and environments work.

cd ./TabSurvey

SKLEARN_ENV="sklearn"
GBDT_ENV="gbdt"
TORCH_ENV="torch"
KERAS_ENV="tensorflow"

# all datasets should be in this folder. the dataset folder should be in ${DATASET_BASE_DIR}/<dataset-name>
DATASET_BASE_DIR=./datasets

##########################################################
# define lists of datasets and models to evaluate them on

MODELS_ENVS=(
  "DANet:$torch"
  )

CONFIG_FILE=tabzilla_experiment_config.yml



DATASETS=(
 openml__APSFailure__168868
 openml__Census-Income__168340
 openml__Fashion-MNIST__146825
 openml__MiniBooNE__168335
 openml__adult__7592
 openml__airlines__189354
 openml__albert__189356
 openml__chess__3952
 openml__connect-4__146195
 openml__electricity__219
 openml__higgs__146606
 openml__jannis__168330
 openml__jungle_chess_2pcs_raw_endgame_complete__167119
 openml__kropt__2076
 openml__ldpa__9974
 openml__mnist_784__3573
 openml__nomao__9977
 openml__numerai28.6__167120
 openml__poker-hand__9890
 openml__skin-segmentation__9965
 openml__volkert__168331
 openml__walking-activity__9945
 openml__CIFAR_10__167124
 openml__Devnagari-Script__167121
 openml__covertype__7593
 openml__guillermo__168337
 openml__helena__168329
 openml__riccardo__168338
 openml__shuttle__146212
)

# conda init bash
eval "$(conda shell.bash hook)"

for dataset in "${DATASETS[@]}"; do
  printf "\n|----------------------------------------------------------------------------\n"
  printf "| starting dataset ${dataset}\n"
  for model_env in "${MODELS_ENVS[@]}"; do

      model="${model_env%%:*}"
      env="${model_env##*:}"

      printf "\n||----------------------------------------------------------------------------\n"
      printf '|| Training %s on dataset %s in env %s\n\n' "$model" "$dataset" "$env"

      conda activate ${env}

      dataset_dir=${DATASET_BASE_DIR}/${dataset}
      python tabzilla_experiment.py --experiment_config ${CONFIG_FILE} --dataset_dir ${dataset_dir} --model_name ${model}

      # zip results into a new directory, and remove unzipped results
      zip -r results_${dataset}_${model}.zip ./results
      rm -r ./results

      conda deactivate

  done
done
