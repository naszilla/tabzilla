#!/bin/bash

# small test script to make sure that TabSurvey code and environments work.

cd ./TabSurvey

N_TRIALS=2
EPOCHS=3

SKLEARN_ENV="sklearn"
GBDT_ENV="gbdt"
TORCH_ENV="torch"
KERAS_ENV="tensorflow"

declare -A MODELS
MODELS=( ["LinearModel"]=$SKLEARN_ENV
         ["KNN"]=$SKLEARN_ENV
         ["DecisionTree"]=$SKLEARN_ENV
          )

CONFIGS=( "config/adult.yml"
          "config/california_housing.yml"
          )

# conda init bash
eval "$(conda shell.bash hook)"

for config in "${CONFIGS[@]}"; do

  for model in "${!MODELS[@]}"; do
    printf "\n\n----------------------------------------------------------------------------\n"
    printf 'Training %s with %s in env %s\n\n' "$model" "$config" "${MODELS[$model]}"

    conda activate "${MODELS[$model]}"

    python tabzilla_experiment.py --config "$config" --model_name "$model" --n_trials $N_TRIALS --epochs $EPOCHS

    conda deactivate

  done

done
