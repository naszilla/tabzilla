# Overview

...

# Python Environments

Hm. Yikes.


# Running Tabzilla Experiments

We modified the TabSurvey code in order to run experiments to generate results for our meta-learning tasks. The script [`TabSurvey/tabzilla_experiment.py`](TabSurvey/tabzilla_experiment.py) runs these experiments (this is adapted from the script [`TabSurvey/train.py`](TabSurvey/train.py)).

Similar to `test.py`, this script writes a database of various results from each train/test cycle, which are recorded and written via [optuna](https://optuna.org/).

## `TabSurvey/tabzilla_experiment.py`

Each ccall to `tabzilla_experiment.py` runs a hyperparameter search for a single algorithm on a single dataset. There are three inputs to this script: the dataset and general parameters (including hyperparameter search params) are passed using their own yml config files; the algorihtm name is passed as a string. 

The three inputs are:
- `--dataset_config`: a yml config file specifying the dataset (see section "Datasets" below)
- `--general_config`: a yml config file specifying general parameters of the experiment (see section "General Parameters" below)
- `--model_name`: a string indicating the model to evaluate. The list of models is imported from `TabSurvey.models.all_models`.

## General Parameters

General parameters for each experiment are read from a yml config file, by the parser returned by [`TabSurvey.tabzilla_utils.get_general_parser`](TabSurvey/tabzilla_utils.py). Below is a description of each of the general parameters read by this parser. An example config file can be found in: [TabSurvey/tabsurvey_search_config.yml](TabSurvey/tabsurvey_search_config.yml).

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


# Datasets

Datasets are handled using the class [`TabSurvey.tabzilla_datasets.TabularDataset`](TabSurvey.tabzilla_datasets.py); all datasets are accessed using an instance of this class. Each dataset is initialized using a function with the decorator `dataset_preprocessor` defined in [`TabSurvey/tabzilla_data_preprocessing.py`](TabSurvey/tabzilla_data_preprocessing.py). Each of these functions is accessed through function `preprocess_dataset()`, which returns any defined datasets by name. For example, the following code will return a `TabularDataset` object representing the `CaliforniaHousing` dataset, and will write it to a local directory unless it already has been written:

```python
from TabSurvey.tabzilla_data_preprocessing import preprocess_dataset

dataset = preprocess_dataset("CaliforniaHousing", overwrite=False)
```

Calling function `preprocess_dataset()` will write a local copy of the dataset (flag `overwrite`) determines whether the dataset will be rewritten if it already exists. It is not necessary to write datasets to file to run experiments (they can just live in memory), however find it helpful to write dataset files for bookkeeping. Once a dataset is preprocessed and written to a local directory, it can be read directly into a `TabularDataset` object.

Calling `tabzilla_data_preprocessing.py` as a script will preprocess selected datasets, writing them to local directories. For example, the following command:

```bash
> python tabzilla_data_preprocessing.py --dataset_name CaliforniaHousing
```

will preprocess and write the `CaliforniaHousing` dataset to local directory `tabzilla/TabSurvey/datasets/CaliforniaHousing`. 

## Reading Preprocessed Datasets

Once a dataset has been preprocessed, as in the above example, it can be read directly into a `TabularDataset` object. For example, if we preprocess `CaliforniaHousing` as shown above, then the following code will read this dataset:

```python
from TabSurvey.tabzilla_datasets import TabularDataset
from pathlib import Path

dataset = TabularDataset.read(Path("tabzilla/TabSurvey/datasets/CaliforniaHousing"))
```

## Adding New Datasets

To add a new dataset, you need to add a new function to [`TabSurvey/tabzilla_data_preprocessing.py`](TabSurvey/tabzilla_data_preprocessing.py), which defines all information about the dataset. This function needs to use the decorator `dataset_preproccessor`. Below is an example:

```python

@dataset_preprocessor("ExampleDataset", target_encode=True)
def preprocess_covertype(dataset_name):

    X = np.array(
      []
    )

    # a list of indices of the categorical and binary features. all other features are assumed to be numerical.
    cat_idx = []

    # can be "binary" "
    target_type = "binary"
    ...

    # TBD

    return dataset
```
