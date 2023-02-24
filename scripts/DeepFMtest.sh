#!/bin/bash

# small test script to make sure that TabSurvey code and environments work.

cd /home/shared/tabzilla/TabSurvey

ENV_NAME=torch

# name of the model/algorithm
MODEL_NAME=DeepFM

# all datasets should be in this folder. the dataset folder should be in ${DATASET_BASE_DIR}/<dataset-name>
DATASET_BASE_DIR=./datasets

##########################################################
# define lists of datasets and models to evaluate them on

# MODELS_ENVS=(
#   "LinearModel:$SKLEARN_ENV"
#   "KNN:$SKLEARN_ENV"
#   "DecisionTree:$SKLEARN_ENV"
#   )

CONFIG_FILE=tabzilla_experiment_config.yml


DATASETS=(
  openml__Fashion-MNIST__146825
  openml__GesturePhaseSegmentationProcessed__14969
  openml__Internet-Advertisements__167125
  openml__JapaneseVowels__3510
  openml__LED-display-domain-7digit__125921
  openml__MiceProtein__146800
  openml__airlines__189354
  openml__albert__189356
  openml__analcatdata_authorship__3549
  openml__analcatdata_dmft__3560
  openml__anneal__2867
  openml__arrhythmia__5
  openml__artificial-characters__14964
  openml__audiology__7
  openml__autos__9
  openml__balance-scale__11
  openml__car-evaluation__146192
  openml__car__146821
  openml__cardiotocography__9979
  openml__chess__3952
  openml__cjs__14967
  openml__cmc__23
  openml__cnae-9__9981
  openml__collins__3567
  openml__connect-4__146195
  openml__dermatology__35
  openml__dilbert__168909
  openml__dna__167140
  openml__ecoli__145977
  openml__eucalyptus__2079
  openml__eye_movements__3897
  openml__fabert__168910
  openml__first-order-theorem-proving__9985
  openml__fl2000__3566
  openml__gas-drift-different-concentrations__9987
  openml__gas-drift__9986
  openml__glass__40
  openml__har__14970
  openml__hayes-roth__146063
  openml__iris__59
  openml__jannis__168330
  openml__jungle_chess_2pcs_raw_endgame_complete__167119
  openml__kropt__2076
  openml__ldpa__9974
  openml__letter__6
  openml__libras__360948
  openml__lung-cancer__146024
  openml__lymph__10
  openml__mfeat-factors__12
  openml__mfeat-fourier__14
  openml__mfeat-karhunen__16
  openml__mfeat-morphological__18
  openml__mfeat-pixel__146824
  openml__mfeat-zernike__22
  openml__mnist_784__3573
  openml__nursery__9892
  openml__one-hundred-plants-texture__9956
  openml__optdigits__28
  openml__page-blocks__30
  openml__poker-hand__9890
  openml__primary-tumor__146032
  openml__satimage__2074
  openml__segment__146822
  openml__semeion__9964
  openml__soybean__41
  openml__splice__45
  openml__steel-plates-fault__146817
  openml__synthetic_control__3512
  openml__tae__47
  openml__texture__125922
  openml__vehicle__53
  openml__volkert__168331
  openml__vowel__3022
  openml__walking-activity__9945
  openml__wall-robot-navigation__9960
  openml__yeast__145793
  openml__isolet__3481
  openml__pendigits__32
  openml__robert__168332
  openml__solar-flare__2068
  openml__CIFAR_10__167124
  openml__Devnagari-Script__167121
  openml__covertype__7593
  openml__helena__168329
  openml__shuttle__146212
)

# conda init bash
eval "$(conda shell.bash hook)"

for DATASET_NAME in ${DATASETS[@]}; do
    printf "\n|----------------------------------------------------------------------------\n"
    printf "| starting dataset ${dataset}\n"
    printf '|| Training %s on dataset %s in env %s\n\n' "$model" "$DATASET_NAME" "$env"

    conda activate ${ENV_NAME}

    DATSET_DIR=${DATASET_BASE_DIR}/${DATASET_NAME}
    python tabzilla_experiment.py --experiment_config ${CONFIG_FILE} --dataset_dir ${DATSET_DIR} --model_name ${MODEL_NAME}

    # zip results into a new directory, and remove unzipped results
    zip -r results_${dataset}_${model}.zip ./results
    rm -r ./results

    conda deactivate
done
