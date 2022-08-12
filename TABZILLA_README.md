# Overview

...

# Python Environments

This repository uses four conda python environments, which are also defined in the TabSurvey dockerfiles. We need to use different environments because some algorithms have different requirements. These four environments are specified in files created using command `conda env export`.

The name of each environment, and their specification file are:
- `sklearn`: [`conda_envs/sklearn.yml`](conda_envs/sklearn.yml)
- `gbdt`: [`conda_envs/gbdt.yml`](conda_envs/gbdt.yml)
- `torch`: [`conda_envs/torch.yml`](conda_envs/torch.yml)
- `tensorflow`: [`conda_envs/tensorflow.yml`](conda_envs/tensorflow.yml)

Each of these four environments can be created using the command `conda env create`:

```bash
conda env create -f ./conda_envs/sklearn.yml
conda env create -f ./conda_envs/gbdt.yml
conda env create -f ./conda_envs/torch.yml
conda env create -f ./conda_envs/tensorflow.yml
```

Different algorithms require different environments. (TBD)


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

Datasets are handled using the class [`TabSurvey.tabzilla_datasets.TabularDataset`](TabSurvey.tabzilla_datasets.py); all datasets are accessed using an instance of this class. Each dataset is initialized using a function with the decorator `dataset_preprocessor` defined in [`TabSurvey/tabzilla_data_preprocessor_utils.py`](TabSurvey/tabzilla_data_preprocessor_utils.py). Each of these functions is accessed through function `preprocess_dataset()`, which returns any defined datasets by name. For example, the following code will return a `TabularDataset` object representing the `CaliforniaHousing` dataset, and will write it to a local directory unless it already has been written:

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

Currently, there are two main procedures to add datasets: one for OpenML datasets, and one for more general datasets. Whenever possible, you should use the OpenML version of the dataset, since it will result in a more seamless process.

### General datasets

To add a new dataset, you need to add a new function to [`TabSurvey/tabzilla_data_preprocessors.py`](TabSurvey/tabzilla_data_preprocessor_utils.py), which defines all information about the dataset. This function needs to use the decorator `dataset_preproccessor`.

Further description of the decorator and its flags TBD.

Below is an example:

```python

@dataset_preprocessor(preprocessor_dict, "ExampleDataset", target_encode=True)
def preprocess_covertype():

    X = np.array(
      []
    )

    # a list of indices of the categorical and binary features. all other features are assumed to be numerical.
    cat_idx = []

    # can be "binary" "
    target_type = "binary"
    ...

    # TBD

    return {
        "X": X,
        "y": y,
        "cat_idx": [],
        "target_type": "classification",
        "num_classes": 7
    }

```

### OpenML datasets
OpenML datasets need to be added under [`TabSurvey/tabzilla_data_preprocessors_openml.py`](TabSurvey/tabzilla_data_preprocessors_openml.py).

OpenML distinguishes tasks from datasets, where tasks are specific prediction tasks associated with a dataset. For our purposes, we will be using OpenML tasks to obtain datasets for training and evaluation.

We use the OpenML API. [Here](https://openml.github.io/openml-python/develop/examples/30_extended/tasks_tutorial.html) is a tutorial on OpenML tasks, including listing tasks according to a series of filters. We will use the benchmark suites as a start, especially the [OpenML-CC18](https://openml.github.io/openml-python/develop/examples/20_basic/simple_suites_tutorial.html#openml-cc18).

#### Step 1: Identifying the dataset

The first step is identifying the OpenML task ID. This can either be obtained by [browsing the lists of OpenML tasks](https://openml.github.io/openml-python/develop/examples/30_extended/tasks_tutorial.html#listing-tasks) and fetching a promising one, searching for a specific dataset within OpenML (e.g. California Housing), or using one of the benchmark suites.

Note that we are focusing on either regression tasks or supervised classification tasks, so please ensure whatever OpenML task you look at belongs to these task types. Furthermore, please ensure the evaluation procedure for the task is "10-fold Crossvalidation". If this is not the case, and you believe the dataset is worth adding to our repo, please let the rest of the team know (this might require either modifying the code or using the procedure for general datasets).


#### Step 2: Manual Inspection

Once you have found the task id for a dataset, the next step is to inspect the dataset. For that, you can use the following piece of code:

```python
import openml
openml_task_id = 361089 # Your task ID goes here
task = openml.tasks.get_task(task_id=openml_task_id)
dataset = task.get_dataset()
X, y, categorical_indicator, col_names = dataset.get_data(
    dataset_format='dataframe',
    target=task.target_name,
)
```

`X` represents the features (as a Pandas dataframe), `y` represents the target, `categorical_indicator` is a list of booleans for each column of X that are true iff the corresponding column is categorical, and `col_names` is the list of column names for `X`.

Please perform the following checks:
1. `task.task_type` is `'Supervised Regression'` or `'Supervised Classification'`.
2. Verify that the `categorical_indicator` has all correct entries by manual inspection of the data. If you find any mislabeled entries, make a note of mislabeled numerical or categorical columns.
3. If the dataset has missing values, please make a note of it (this is not necessarily a deal-breaker, but might require modifications to the code).
4. `task.estimation_procedure['type']` should be `'crossvalidation'` with `task.estimation_procedure['parameters'] = {'number_repeats': '1', 'number_folds': '10', 'percentage': ''}`. If this does not match, please let the team know since this might require modifications to the code.


#### Step 3: Adding the dataset
If the dataset passes all of these checks (which should be the case for the curated benchmarks), you have two options to add the dataset:
1. Adding the task ID to `openml_easy_import_list.txt`
2. Adding the task data as a dictionary in the list `openml_tasks` under `tabzilla_preprocessors_openml.py`.

Option 1 is suited for quick addition of a dataset that has no problems or additional cleaning required. Simply add the task ID as a new line in `openml_easy_import_list.txt`. The dataset will be given an automatic name with the format `f"openml_{OPENML_DATASET_NAME}"`. You can find the `OPENML_DATASET_NAME` dataset name using `task.get_dataset().name`.

For some datasets, you might need to use Option 2. In particular, Option 2 lets you specify the following for any dataset:
1. `"openml_task_id"` (required): the OpenML task ID 
2. `"dataset_name"` (optional): specify a manual dataset name (if you want something other than `f"openml_{DATASET_NAME}"`). Specifying this can result in faster execution of the pre-processing script.
3. `"target_type"` (optional): The target type can be automatically determined by the code based on the OpenML task metadata, but you can force the `"target_type"` by specifying this attribute. The options are: `"regression"`, `"binary"`, and `"classification"`.
4. `"force_cat_features"` (optional): list of strings specifying column names for columns that will be forced to be treated as categorical. Use if you found categorical columns incorrectly labeled in `categorical_indicator`. You only need to specify categorical columns which were incorrectly labeled (not all of them).
5. `"force_num_features"` (optional): list of strings specifying column names for columns that will be forced to be treated as numerical. Use if you found numerical columns incorrectly labeled in `categorical_indicator`. You only need to specify numerical columns which were incorrectly labeled (not all of them).

Here is an example:

```python
    {
        "openml_task_id": 2071,
        "dataset_name": "openml_adult", # Can be explicitly specified for faster execution
        "target_type": "binary", # Does not need to be explicitly specified, but can be
        "force_cat_features": ["workclass", "education"], # Example (these are not needed in this case)
        "force_num_features": ["fnlwgt", "education-num"], # Example (these are not needed in this case)
    }
```

You do not need to provide all of the fields. Once you are done, add the dictionary entry to `openml_tasks` under `tabzilla_preprocessors_openml.py`.


#### Step 4: Testing pre-processing on the dataset

The final step is running pre-processing on the dataset. From `TabSurvey`, run the following:

```bash
> python tabzilla_data_preprocessing.py --dataset_name YOUR_DATASET_NAME
```

(If you used the easy import option and you do not know the dataset name, you can use `task.get_dataset().name` to find the OpenML dataset name. Use that name with the prefix `openml_`.)

This should output a folder under `TabSurvey/datasets/YOUR_DATASET_NAME` with files `metadata.json`, `split_indeces.npy.gz`, `X.npy.gz`, and `y.npy.gz`. Open `metadata.json` and check that the metadata corresponds to what you expect (especially `target_type`).