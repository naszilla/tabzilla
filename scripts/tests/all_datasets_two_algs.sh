#!/bin/bash

# load functions
source ../utils.sh

##############################
# begin: EXPERIMENT PARAMETERS

# you can produce this model in python using:
# > from tabzilla_alg_handler import ALL_MODELS
# > for k,v in ALL_MODELS.items(): print(f"{k}:{v[0]}")
MODELS_ENVS=(
  KNN:$SKLEARN_ENV
  CatBoost:$GBDT_ENV
  )

# to generate this list, run in python:
# > from tabzilla_data_preprocessing import build_preprocessors_dict
# > preprocessors = build_preprocessors_dict()
# > for k, v in preprocessors.items(): print(k)
DATASETS=(
    openml__sick__3021
    openml__kr-vs-kp__3
    openml__letter__6
    openml__balance-scale__11
    openml__mfeat-factors__12
    openml__mfeat-fourier__14
    openml__breast-w__15
    openml__mfeat-karhunen__16
    openml__mfeat-morphological__18
    openml__mfeat-zernike__22
    openml__cmc__23
    openml__optdigits__28
    openml__credit-approval__29
    openml__credit-g__31
    openml__pendigits__32
    openml__diabetes__37
    openml__spambase__43
    openml__splice__45
    openml__tic-tac-toe__49
    openml__vehicle__53
    openml__electricity__219
    openml__satimage__2074
    openml__eucalyptus__2079
    openml__vowel__3022
    openml__isolet__3481
    openml__analcatdata_authorship__3549
    openml__analcatdata_dmft__3560
    openml__mnist_784__3573
    openml__pc4__3902
    openml__pc3__3903
    openml__jm1__3904
    openml__kc2__3913
    openml__kc1__3917
    openml__pc1__3918
    openml__adult__7592
    openml__Bioresponse__9910
    openml__wdbc__9946
    openml__phoneme__9952
    openml__qsar-biodeg__9957
    openml__wall-robot-navigation__9960
    openml__semeion__9964
    openml__ilpd__9971
    openml__madelon__9976
    openml__nomao__9977
    openml__ozone-level-8hr__9978
    openml__cnae-9__9981
    openml__first-order-theorem-proving__9985
    openml__banknote-authentication__10093
    openml__blood-transfusion-service-center__10101
    openml__PhishingWebsites__14952
    openml__cylinder-bands__14954
    openml__bank-marketing__14965
    openml__GesturePhaseSegmentationProcessed__14969
    openml__har__14970
    openml__dresses-sales__125920
    openml__texture__125922
    openml__connect-4__146195
    openml__MiceProtein__146800
    openml__steel-plates-fault__146817
    openml__climate-model-simulation-crashes__146819
    openml__wilt__146820
    openml__car__146821
    openml__segment__146822
    openml__mfeat-pixel__146824
    openml__Fashion-MNIST__146825
    openml__jungle_chess_2pcs_raw_endgame_complete__167119
    openml__numerai28.6__167120
    openml__Devnagari-Script__167121
    openml__CIFAR_10__167124
    openml__Internet-Advertisements__167125
    openml__dna__167140
    openml__churn__167141
    openml__california__361089
    openml__covertype__7593
    openml__adult-census__3953
    openml__Amazon_employee_access__34539
    openml__shuttle__146212
    openml__higgs__146606
    openml__Australian__146818
    openml__helena__168329
    openml__jannis__168330
    openml__volkert__168331
    openml__robert__168332
    openml__MiniBooNE__168335
    openml__guillermo__168337
    openml__riccardo__168338
    openml__APSFailure__168868
    openml__christine__168908
    openml__dilbert__168909
    openml__fabert__168910
    openml__jasmine__168911
    openml__sylvine__168912
    openml__airlines__189354
    openml__dionis__189355
    openml__albert__189356
)

# base name for the gcloud instances
instance_base=all-datasets

# experiment name (will be appended to results files)
experiment_name=all-datasets

# maximum number of experiments (background processes) that can be running
MAX_PROCESSES=10

# config file
config_file=/home/shared/tabzilla/TabSurvey/tabzilla_experiment_config_test.yml

# end: EXPERIMENT PARAMETERS
############################

####################
# begin: bookkeeping

# make a log directory
mkdir ${PWD}/logs
LOG_DIR=${PWD}/logs

###################
# set trap
# this will sync the log files, and delete all instances

trap "sync_logs ${LOG_DIR} ${experiment_name}; delete_instances" EXIT
INSTANCE_LIST=()

# end: bookkeeping
##################


#################
# run experiments

num_experiments=0
for i in ${!MODELS_ENVS[@]};
do
  for j in ${!DATASETS[@]};
  do
    model_env="${MODELS_ENVS[i]}"
    model="${model_env%%:*}"
    env="${model_env##*:}"

    instance_name=${instance_base}-${i}-${j}

    # args:
    # $1 = model name
    # $2 = dataset name
    # $3 = env name
    # $4 = instance name
    # $5 = experiment name
    echo "MODEL_ENV: ${model_env}"
    echo "MODEL: ${model}"
    echo "ENV: ${env}"
    echo "DATASET: ${DATASETS[j]}"
    echo "EXPERIMENT_NAME: ${experiment_name}"

    run_experiment "${model}" ${DATASETS[j]} ${env} ${instance_base}-${i}-${j} ${experiment_name} ${config_file} >> ${LOG_DIR}/log_${i}_${j}_$(date +"%m%d%y_%H%M%S").txt 2>&1 &
    num_experiments=$((num_experiments + 1))

    # add instance name to the instance list
    INSTANCE_LIST+=("${instance_name}")

    echo "launched instance ${instance_base}-${i}-${j}. (job number ${num_experiments})"
    sleep 1

    # if we have started MAX_PROCESSES experiments, wait for them to finish
    wait_until_processes_finish $MAX_PROCESSES

  done
done

echo "still waiting for processes to finish..."
wait
echo "done."