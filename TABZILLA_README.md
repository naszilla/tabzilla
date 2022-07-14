# Overview

...

## Running Tabzilla Experiments

We modified the TabSurvey code in order to run experiments to generate results for our meta-learning tasks. The script [`TabSurvey/tabzilla_experiment.py`](TabSurvey/tabzilla_experiment.py) runs these experiments (this is adapted from the script [`TabSurvey/train.py`](TabSurvey/train.py)).

Similar to `test.py`, this script writes a database of various results from each train/test cycle, which are recorded and written via [optuna](https://optuna.org/).

### `TabSurvey/tabzilla_experiment.py`

Each ccall to `tabzilla_experiment.py` runs a hyperparameter search for a single algorithm on a single dataset. There are three inputs to this script: the dataset and general parameters (including hyperparameter search params) are passed using their own yml config files; the algorihtm name is passed as a string. 

The three inputs are:
- `--dataset_config`: a yml config file specifying the dataset (see section "Datasets" below)
- `--general_config`: a yml config file specifying general parameters of the experiment (see section "General Parameters" below)
- `--model_name`: a string indicating the model to evaluate. The list of models is imported from `TabSurvey.models.all_models`.

### General Parameters

General parameters for each experiment are read from a yml config file, by the parser returned by [`TabSurvey.tabzilla_utils.get_general_parser`](TabSurvey/tabzilla_utils.py). Below is a description of each of the general parameters read by this parser. An example config file can be found in: [TabSurvey/tabzilla_config_library/general.yml](TabSurvey/tabzilla_config_library/general.yml).

**General config parameters**
```
  --use_gpu             Set to true if GPU is available (default: False)
  --gpu_ids GPU_IDS     IDs of the GPUs used when data_parallel is true (default: None)
  --data_parallel       Distribute the training over multiple GPUs (default: False)
  --optimize_hyperparameters
                        Search for the best hyperparameters (default: False)
  --n_trials N_TRIALS   Number of trials for the hyperparameter optimization (default: 100)
  --num_splits NUM_SPLITS
                        Number of splits done for cross validation (default: 5)
  --shuffle             Shuffle data during cross-validation (default: False)
  --seed SEED           Seed for KFold initialization. (default: 123)
  --batch_size BATCH_SIZE
                        Batch size used for training (default: 128)
  --val_batch_size VAL_BATCH_SIZE
                        Batch size used for training and testing (default: 128)
  --early_stopping_rounds EARLY_STOPPING_ROUNDS
                        Number of rounds before early stopping applies. (default: 20)
  --epochs EPOCHS       Max number of epochs to train. (default: 1000)
  --logging_period LOGGING_PERIOD
                        Number of iteration after which validation is printed. (default: 100)
```


### Datasets

Each dataset is specified by its own yml config file, similar to the way that datasets are specified in the TabSurvey codebase. Parameters from the dataset config file are read using the parser returned by [`TabSurvey.tabzilla_utils.get_dataset_parser`](`TabSurvey/tabzilla_utils.py`). Below is a description of each of the dataset parameters read by this parser. We store config files for all tabzilla experiments in: [TabSurvey/tabzilla_config_library/datasets](TabSurvey/tabzilla_config_library/datasets).

**Dataset config parameters**
```
  --dataset DATASET     Name of the dataset that will be used (default: None)
  --objective {regression,classification,binary}
                        Set the type of the task (default: regression)
  --direction {minimize,maximize}
                        Direction of optimization. (default: minimize)
  --num_features NUM_FEATURES
                        Set the total number of features. (default: None)
  --num_classes NUM_CLASSES
                        Set the number of classes in a classification task. (default: 1)
  --cat_idx CAT_IDX     Indices of the categorical features (default: None)
  --cat_dims CAT_DIMS   Cardinality of the categorical features (is set automatically, when the load_data function is used. (default: None)
  --scale               Normalize input data. (default: False)
  --target_encode       Encode the targets that they start at 0. (0, 1, 2,...) (default: False)
  --one_hot_encode      OneHotEncode the categorical features (default: False)
```