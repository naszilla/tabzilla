<br/>
<p align="center"><img src="img/tabzilla_logo.png" width=700 /></p>

----
![Crates.io](https://img.shields.io/crates/l/Ap?color=orange)


`TabZilla` is a framework which provides the functionality to compare many different tabular algorithms across a large, diverse set of tabular datasets, as well as to determine dataset properties associated with the performance of certain algorithms and algorithm families.

See our NeurIPS 2023 Datasets and Benchmarks paper at [https://arxiv.org/abs/2305.02997](https://arxiv.org/abs/2305.02997).


# Overview

This codebase extends the excellent public repository [TabSurvey](https://github.com/kathrinse/TabSurvey), by Vadim Borisov, Tobias Leemann, Kathrin Seßler, Johannes Haug, Martin Pawelczyk, and Gjergji Kasneci.

The `TabZilla` codebase implements a wide range of machine learning algorithms and tabular datasets, using a common interface. This allows users to train and evaluate different many different algorithms on many different datasets using the same procedures, with the same dataset splits---in a true "apples-to-apples" comparison.

This codebase has two primary components:
1. **Running Experiments:** In this codebase, an "experiment" refers to a running a single algorithm on a single dataset. An experiment can run multiple hyperparameter samples for the algorithm, and each hyperparameter is trained and evaluated for each dataset split. See section [Running TabZilla Experiments](#running-tabzilla-experiments) for details.
2. **Extracting Dataset Metafeature:** Each dataset can be represented by a set of numerical "metafeatures". Our codebase uses [PyMFE](https://github.com/ealcobaca/pymfe) to calculate metafeatures for each dataset fold. These metafeatures can be used for analyzing what properties of a dataset make a certain algorithm better-suited to perform well, which is one focus of our paper. See section [Metafeature Extraction](#metafeature-extraction) for details.

Adding new datasets and algorithms to this codebase is fairly easy. All datasets implemented in this repo are from [OpenML](https://www.openml.org/), and adding new OpenML datasets is especially easy (see section [Adding New Datasets](#adding-new-datasets)). Adding new algorithms requires an sklearn-style interface (see section [Implementing New Models](#implementing-new-models)). If a new algorithm requires a new python environment, this new environment can be added to our codebase pretty easily as well (see section [Preparing Python Environments](#Python)).


## Table of Contents
1. [Documentation](#documentation)
2. [Preparing Python Environments](#preparing-a-python-environment)
3. [Running TabZilla Experiments](#running-tabzilla-experiments)
    1. [Experiment Script](#experiment-script)
    2. [Experiment Config Parser](#experiment-config-parser)
    3. [Running Individual Experiments](#running-individual-experiments)
4. [Datasets](#datasets)
    1. [Dataset Class and Preprocessing](#dataset-class-and-preprocessing) 
    2. [Reading Preprocessed Datasets](#reading-preprocessed-datasets)
    3. [Adding New Datasets](#adding-new-datasets)
6. [Metafeature Extraction](#metafeature-extraction)
7. [Implementing New Models](#implementing-new-models)
8. [Unit Tests](#unit-tests)

# Documentation
Here, we describe our dataset documentation. All of this information is also available in our paper.
- [Author Responsibility](docs/AUTHOR_RESPONSIBILITY.md)
- [Code of Conduct](docs/CODE_OF_CONDUCT.md)
- [Contributing](docs/CONTRIBUTING.md)
- [Datasheet for TabZilla](docs/DATASHEET.md)
- [Maintenance Plan](docs/MAINTENANCE_PLAN.md)

# Preparing a Python Environment

The core functionality of TabZilla requires only three packages: [`optuna`](https://pypi.org/project/optuna/), [`scikit-learn`](https://pypi.org/project/scikit-learn/), [`openml`](https://pypi.org/project/openml) and [`configargparse`](https://pypi.org/project/ConfigArgParse/). Below we give instructions to build a single python 3.10 environment that can run all 23 algorithms used in this study, as well as dealing with dataset preparation and featurization. Depending on your needs, you might not need all packages here.

### Creating a TabZilla virtual environment with `venv`

We recommend using `venv` and `pip` to create an environment, since some ML algorithms require specific package versions. You can use the following instructions:

1. Install python 3.10 (see these recommendations specific to [Mac](https://formulae.brew.sh/formula/python@3.10), for Windows and Linux see the [python site](https://www.python.org/downloads/release/python-31012/)). We use python 3.10 because a few algorithms currently require it. Make sure you can see the python 3.10 install, for example like this:

```
> python3.10 --version

Python 3.10.12
```

2. Create a virtual environment with `venv` called "tabzilla" (or whatever you want to call it), using this version of python. Change the name at the end of the path (tabzilla) if you want this virtual environment named differently. This will create a virtual environment in your current directory called "tabzilla" (Mac and Linux only):

```
> python3.10 -m venv ./tabzilla
```

and activate the virtual environment:
```
> source /home/virtual-environments/tabzilla/bin/activate
```

3. Install all tabzilla dependencies using the pip requirements file [`TabZilla/pip_requirements.txt`](TabZilla/pip_requirements.txt):

```
> pip install -r ./pip_requirements.txt
```

4. Test this python environment using TabZilla unittests. All tests should pass:

```
> python -m unittest unittests.test_experiments 
```

and test a specific algorithm using `unittests.test_alg` **without** using the unittest module. For example, to test algorithm "rtdl_MLP", run:

```
> python -m unittests.test_alg rtdl_MLP
```

# Running TabZilla Experiments

The script [`TabZilla/tabzilla_experiment.py`](TabZilla/tabzilla_experiment.py) runs an "experiment", which trains and tests a single algorithm with a single dataset. This experiment can test multiple hyperparameter sets for the algorithm; for each hyperparameter sample, we train & evaluate on each dataset split.

## Experiment Script

Each call to `tabzilla_experiment.py` runs a hyperparameter search for a single algorithm on a single dataset. There are three inputs to this script: the dataset and general parameters (including hyperparameter search params) are passed using their own yml config files; the algorithm name is passed as a string. 

The three inputs are:
- `--experiment_config`: a yml config file specifying general parameters of the experiment. Our default config file is here: [`TabZilla/tabzilla_experiment_config.yml`](TabZilla/tabzilla_experiment_config.yml) 
- `--model_name`: a string indicating the model to evaluate. The list of valid model names is the set of keys for dictionary `ALL_MODELS` in file [`TabZilla/tabzilla_alg_handler.py`](TabZilla/tabzilla_alg_handler.py)
- `--dataset_dir`: the directory of the processed dataset to use. This directory should be created 


## Experiment Config Parser

General parameters for each experiment are read from a yml config file, by the parser returned by [`TabZilla.tabzilla_utils.get_general_parser`](TabZilla/tabzilla_utils.py). Below is a description of each of the general parameters read by this parser. For debugging, you can use the example config file here: [TabZilla/tabzilla_experiment_config.yml](TabZilla/tabzilla_experiment_config.yml).

**General config parameters**
```
  --output_dir OUTPUT_DIR
                        directory where experiment results will be written. (default: None)
  --use_gpu             Set to true if GPU is available (default: False)
  --gpu_ids GPU_IDS     IDs of the GPUs used when data_parallel is true (default: None)
  --data_parallel       Distribute the training over multiple GPUs (default: False)
  --n_random_trials N_RANDOM_TRIALS
                        Number of trials of random hyperparameter search to run (default: 10)
  --hparam_seed HPARAM_SEED
                        Random seed for generating random hyperparameters. passed to optuna RandomSampler. (default: 0)
  --n_opt_trials N_OPT_TRIALS
                        Number of trials of hyperparameter optimization to run (default: 10)
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

## Running Individual Experiments

The script [`scripts/test_tabzilla_on_instance.sh`](scripts/test_tabzilla_on_instance.sh) gives an example of a single experiment. That is, running a single algorithm on a single dataset, using parameters specified in an experiment config file. We wrote this script to run experiments on a cloud instance (GCP), but it can be run anywhere as long as all python environments and datasets are present.

# Datasets

**Note:** Our code downloads datasets from [OpenML](https://www.openml.org/), so you will need to install the openml python module. If this code hangs or raises an error when downloading datasets, you may need to create an OpenML account (on their website) and authenticate your local machine in order to download datasets. If you run into any issues, please follow [these installation and authentication instructions](https://openml.github.io/openml-python/main/examples/20_basic/introduction_tutorial.html#sphx-glr-examples-20-basic-introduction-tutorial-py).  

**To download and pre-process all datasets**, use run the following command from the TabZilla folder:

```bash
> python tabzilla_data_preprocessing.py --process_all
```

This will download all datasets, and write a pre-processed version of each
to a local directory `TabZilla/datasets/<dataset name>`.

**To download and pre-process a single dataset**, run the following from the TabZilla folder

```bash
> python tabzilla_data_preprocessing.py --dataset_name <dataset name>
```

For example, the following command will download the dataset "openml__california__361089":

```bash
> python tabzilla_data_preprocessing.py --dataset_name openml__california__361089
```

To print a list of all dataset names that can be passed to this script, run:

```bash
> python tabzilla_data_preprocessing.py --print_dataset_names
```

## Dataset Class and Preprocessing

Datasets are handled using the class [`TabZilla.tabzilla_datasets.TabularDataset`](TabZilla/tabzilla_datasets.py); all datasets are accessed using an instance of this class. Each dataset is initialized using a function with the decorator `dataset_preprocessor` defined in [`TabZilla/tabzilla_preprocessor_utils.py`](TabZilla/tabzilla_preprocessor_utils.py). Each of these functions is accessed through function `preprocess_dataset()`, which returns any defined datasets by name. For example, the following code will return a `TabularDataset` object representing the `openml__california__361089` dataset, and will write it to a local directory unless it already has been written:

```python
from TabZilla.tabzilla_data_preprocessing import preprocess_dataset

dataset = preprocess_dataset("openml__california__361089", overwrite=False)
```

Calling function `preprocess_dataset()` will write a local copy of the dataset (flag `overwrite`) determines whether the dataset will be rewritten if it already exists. It is not necessary to write datasets to file to run experiments (they can just live in memory), however find it helpful to write dataset files for bookkeeping. Once a dataset is preprocessed and written to a local directory, it can be read directly into a `TabularDataset` object.

## Reading Preprocessed Datasets

Once a dataset has been preprocessed, as in the above example, it can be read directly into a `TabularDataset` object. For example, if we preprocess `CaliforniaHousing` as shown above, then the following code will read this dataset:

```python
from TabZilla.tabzilla_datasets import TabularDataset
from pathlib import Path

dataset = TabularDataset.read(Path("tabzilla/TabZilla/datasets/openml__california__361089"))
```

## Adding New Datasets

Currently, there are two main procedures to add datasets: one for OpenML datasets, and one for more general datasets. Whenever possible, you should use the OpenML version of the dataset, since it will result in a more seamless process.

### General (non-OpenML) datasets

To add a new dataset, you need to add a new function to [`TabZilla/tabzilla_preprocessors.py`](TabZilla/tabzilla_preprocessor_utils.py), which defines all information about the dataset. This function needs to use the decorator `dataset_preproccessor`, and is invoked through `tabzilla_data_preprocessing.py`.

In general, the function must take no arguments, and it must return a dictionary with keys used to initialize a `TabularDataset` object. The following keys are required (since they are required by the constructor):
1. `X`: features, as numpy array of shape `(n_examples, n_features)`
2. `y`: labels, as numpy array of shape `(n_samples,)`
3. `cat_idx`: sorted list of indeces of categorical columns in `X`.
4. `target_type`: one of `"regression"`, `"binary"`, and `"classification"`.
5. `"num_classes"`: number of classes, as an integer. Use 1 for `"regression"` and `"binary"`, and the actual number of classes for `"classification"`.
6. Any other optional arguments that you wish to manually specify to create the `TabularDataset` object (usually not needed, since they are inferred if not automatically detected).

Regarding the decorator `dataset_preproccessor`, it takes in the following arguments:
1. `preprocessor_dict`: set to `preprocessor_dict` if adding a pre-processor within `tabzilla_preprocessors.py` (this is used to add an entry to `preprocessor_dict` that will correspond to the new dataset preprocessor).
2. `dataset_name`: unique string name that will be used to refer to the dataset. This name will be used by `tabzilla_data_preprocessing.py` and it will be used in the save location for the dataset.
3. `target_encode` (optional): flag to specify whether to run `y` through a Label Encoder. If not specified, then the Label Encoder will be used iff the `target_type` is `binary` or `classification`.
4. `cat_feature_encode` (optional): flag to indicate whether a Label Encoder should be used on the categorical features. By default, this is set to `True`.
5. `generate_split` (optional): flag to indicate whether to generate a random split (based on a seed) using 10-fold cross validation (as implemented in `split_dataset` in `tabzilla_preprocessor_utils.py`). Defaults to `True`. If set to `False`, you should specify a split using the `split_indeces` entry in the output dictionary of the function.

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
This dataset will be named `"ExampleDataset"`, with Label Encoding being applied to the target and the categorical features, and a default split being generated using `split_dataset`.

Once you have implemented a new dataset, verify that pre-processing runs as expected. From `TabZilla`, run the following:

```bash
> python tabzilla_data_preprocessing.py --dataset_name YOUR_DATASET_NAME
```

This should output a folder under `TabZilla/datasets/YOUR_DATASET_NAME` with files `metadata.json`, `split_indeces.npy.gz`, `X.npy.gz`, and `y.npy.gz`. Open `metadata.json` and check that the metadata corresponds to what you expect.

### OpenML datasets
OpenML datasets need to be added under [`TabZilla/tabzilla_preprocessors_openml.py`](TabZilla/tabzilla_preprocessors_openml.py).

OpenML distinguishes tasks from datasets, where tasks are specific prediction tasks associated with a dataset. For our purposes, we will be using OpenML tasks to obtain datasets for training and evaluation.

We use the OpenML API. [Here](https://openml.github.io/openml-python/develop/examples/30_extended/tasks_tutorial.html) is a tutorial on OpenML tasks, including listing tasks according to a series of filters. A good resource are benchmark suites, such as the [OpenML-CC18](https://openml.github.io/openml-python/develop/examples/20_basic/simple_suites_tutorial.html#openml-cc18). However, note that OpenML-CC18 tasks have already been imported into the repository.

#### Step 1: Identifying the dataset

The first step is identifying the OpenML task ID. This can either be obtained by [browsing the lists of OpenML tasks](https://openml.github.io/openml-python/develop/examples/30_extended/tasks_tutorial.html#listing-tasks) and fetching a promising one, searching for a specific dataset within OpenML (e.g. California Housing), or using one of the benchmark suites.

A convenience function has been added to fetch a dataframe with all relevant OpenML tasks. Call `get_openml_task_metadata` within `tabzilla_preprocessors_openml.py` to obtain a dataframe listing all available tasks, indexed by task ID. The column `in_repo` indicates whether the task has already been added to the repo or not. **Please do not add a task for which there is already a task in the repo that uses the same dataset.**

All datasets currently have the evaluation procedure set to "10-fold Crossvalidation".


#### Step 2: Inspection

Once you have found the task id for a dataset, the next step is to inspect the dataset. For that, run the following from a Python console with `TabZilla` as the working directory:

```python
from tabzilla_preprocessors_openml import inspect_openml_task
inspect_openml_task(YOUR_OPENML_TASK_ID, exploratory_mode=True)
```

The function performs the following checks
1. `task.task_type` is `'Supervised Regression'` or `'Supervised Classification'`.
2. No column is composed completely of missing values. No labels are missing. In addition, the number of missing values is printed out.
3. Categorical columns are correctly identified (see `categorical_indicator` within the function).
4. The estimation procedure is 10-fold cross validation.  Sometimes, a different task might use the same dataset with 10-fold cross validation, so please check for that. If this still does not match, please let the team know since this might require modifications to the code.

If all checks are passed, the output is similar to the following:
```python
inspect_openml_task(7592, exploratory_mode=True)
TASK ID: 7592
Warning: 6465 missing values.
Tests passed!
(Pdb)
```
The debugger is invoked to allow you to inspect the dataset (eliminate the `exploratory_mode` argument to change this behavior).

If, on the other hand, some checks fail, the output is similar to the following:
```python
inspect_openml_task(3021, exploratory_mode=True)
TASK ID: 3021
Warning: 6064 missing values.
Errors found:
Found full null columns: ['TBG']
Mislabeled categorical columns
(Pdb)
```
The debugger is invoked to allow you to see how to rectify the issues. In this case, a column needs to be dropped from the dataset (`"TBG"`).

It is also possible to run the checks on all the tasks of a suite as a batch. You can run:
```python
from tabzilla_preprocessors_openml import check_tasks_from_suite
suite_id = 218 # Example
succeeded, failed = check_tasks_from_suite(suite_id)
```
This function runs `inspect_openml_task` on all the tasks from the specified suite **that have not yet been added to the repository**. `succeeded` contains a list of the task IDs for tasks that passed all of the tests, while `failed` contains the other tasks.

#### Step 3: Adding the dataset
If the dataset passes all of these checks (which should be the case for the curated benchmarks), you have two options to add the dataset:
1. Adding the task ID to `openml_easy_import_list.txt`
2. Adding the task data as a dictionary in the list `openml_tasks` under `tabzilla_preprocessors_openml.py`.

Option 1 is suited for quick addition of a dataset that has no problems or additional cleaning required. Simply add the task ID as a new line in `openml_easy_import_list.txt`. The dataset will be given an automatic name with the format `f"openml__DATASET_NAME__TASK_ID"`. You can find the `OPENML_DATASET_NAME` dataset name using `task.get_dataset().name`.

For some datasets, you might need to use Option 2. In particular, Option 2 lets you specify the following for any dataset:
1. `"openml_task_id"` (required): the OpenML task ID
2. `"target_type"` (optional): The target type can be automatically determined by the code based on the OpenML task metadata, but you can force the `"target_type"` by specifying this attribute. The options are: `"regression"`, `"binary"`, and `"classification"`.
3. `"force_cat_features"` (optional): list of strings specifying column names for columns that will be forced to be treated as categorical. Use if you found categorical columns incorrectly labeled in `categorical_indicator`. You only need to specify categorical columns which were incorrectly labeled (not all of them).
4. `"force_num_features"` (optional): list of strings specifying column names for columns that will be forced to be treated as numerical. Use if you found numerical columns incorrectly labeled in `categorical_indicator`. You only need to specify numerical columns which were incorrectly labeled (not all of them).
5. `"drop_features"` (optional): list of strings specifying column names for columns that will be dropped.

Here is an example:

```python
{
    "openml_task_id": 7592,
    "target_type": "binary", # Does not need to be explicitly specified, but can be
    "force_cat_features": ["workclass", "education"], # Example (these are not needed in this case)
    "force_num_features": ["fnlwgt", "education-num"], # Example (these are not needed in this case)
}
```

You do not need to provide all of the fields. Once you are done, add the dictionary entry to `openml_tasks` under `tabzilla_preprocessors_openml.py`.


#### Step 4: Testing pre-processing on the dataset

The final step is running pre-processing on the dataset. From `TabZilla`, run the following:

```bash
> python tabzilla_data_preprocessing.py --dataset_name YOUR_DATASET_NAME
```

(If you do not know the dataset name, it will the format `f"openml__DATASET_NAME__TASK_ID"`. You can find the `OPENML_DATASET_NAME` dataset name using `task.get_dataset().name`. Alternatively, run the script with the flag `--process_all` instead of the `--dataset_name` flag). 

This should output a folder under `TabZilla/datasets/YOUR_DATASET_NAME` with files `metadata.json`, `split_indeces.npy.gz`, `X.npy.gz`, and `y.npy.gz`. Open `metadata.json` and check that the metadata corresponds to what you expect (especially `target_type`). Note that running the pre-processing also performs the checks within `inspect_openml_task` again, which is particularly useful if you had to make any changes (for Option 2 of OpenML dataset addition). This ensures the final dataset saved to disk passes the checks.

# Metafeature Extraction

The script for extracting metafeatures is provided in [`TabZilla/tabzilla_featurizer.py`](TabZilla/tabzilla_featurizer.py). It uses [PyMFE](https://pymfe.readthedocs.io/en/latest/index.html) to extract metafeatures from the datasets. Note that PyMFE currently does not support regression tasks, so the featurizer will skip regression datasets.

To extract metafeatures, you first need to have the dataset(s) you want to extract metafeatures on disk (follow the instructions from the **Datasets** section for this). Next, run `tabzilla_featurizer.py` (no arguments needed). The script will walk the datasets folder, extract metafeatures for each dataset (that is not a regression task), and write the metafeatures to `metafeatures.csv`. Note that the script saves these metafeatures after each dataset has been processed, so if the script is killed halfway through a dataset, the progress is not lost and only datasets that have not been featurized are processed.

Each row corresponds to one dataset fold. Metafeature columns start with the prefix `f__`.

There are three main settings that control the metafeatures extracted, and they are defined at the top of the script. These are:
1. `groups`: List of groups of metafeatures to extract. The possible values are listed in the comments. In general, we should aim to extract as many metafeatures as possible. However, some metafeature categories can result in expensive computations that run out of memory, so some categories are not currently selected.
2. `summary_funcs`: functions to summarize distributions. The possible values are listed in the comments, and the current list includes all of them.
3. `scoring`: scoring function used for landmarkers. Possible values are listed in the comments.


**It is very important that you use a consistent setting of metafeatures for all datasets**. Extracting metafeatures for some datasets, changing the datasets, and then appending to the same `metafeatures.csv` file is not recommended. It is possible to modify the script so that if entries are added to `groups`, the script only computes the new group of metafeatures for all datasets. However, this behavior has not been implemented, and the current version of the script assumes that the metafeature settings do not change in between runs.

There are a few additional settings that control PyMFE's metafeature extraction process within the script. These are set fixed in the code but can also be modified if needed:
1. `random_state` (used in `MFE` initialization): Set to 0 for reproducibility.
2. `transform_num`: boolean flag used in the `fit` method of the `MFE` object. Setting it to true causes numerical features to be transformed into categorical for metafeatures that can only be computed with categorical features. This behavior is memory-intensive, so it has been disabled, but it also means that some metafeatures that are computed on categorical features will be missing or less reliable for datasets with none or few categorical features, respectively.
3. `transform_cat`: analogous to `transform_num`, for categorical features to be converted into numerical ones. Setting it to `None` disables the behavior, and this is currently done in the script to avoid memory issues. For the different options, see the PyMFE source code and documentation.

Extracting metafeatures can take several days for all datasets, so it is recommended to run the script within a terminal multiplexer such as screen. Paralellization of the script might be desirable in the future, but memory issues might arise with some of the computations if done on a single instance.

# Implementing new models


You can follow the [original TabSurvey readme](TabZilla/TabSurvey_README.md) to implement new models, with the following additions.

For any model supporting multi-class classification, you need to ensure the model follows one of the next two approaches:
1. The model always encodes its output with `args.num_classes` dimension (this is set to 1 for binary classification). In the case of multi-class classification, dimension `i` must match to the value `i` in the labels (which are encoded 0 through `args.num_classes-1` in the output). **Note**: inferring the number of classes from the labels in training may not be sufficient if there are missing labels on the training set (which happens for some datasets), so you must use `args.num_classes` directly.
2. If there is a chance for the prediction probabilities for the model to have less than `args.num_classes` dimension (this can mainly happen if there are missing classes in training for models such as those from `sklearn`) implement a method `get_classes()` that returns the list of the labels corresponding to the dimensions. See [examples here](TabZilla/models/baseline_models.py).


# Unit Tests

The unit tests in [TabZilla/unittests/test_expriments.py](TabZilla/unittests/test_experiments.py) and [TabZilla/unittests/test_alg.py](TabZilla/unittests/test_alg.py) test different algorithms on five datasets using our experiment function.

To run tests for two algorithms (linearmodel and randomforest), run the following from the TabZilla directory:

```
python -m unittests.test_experiments
```

To test a specific algorithm, use `unittests.test_alg`, and pass a single positional argument, the algorithm name:

```
python -m unittests.test_alg <alg_name>
```

**Hint:** To see all available algorithm names, run the file `tabzilla_alg_handler.py` as a script:

```
python -m tabzilla_alg_handler
```

which will print:

```
all algorithms:
LinearModel
KNN
SVM
DecisionTree
...
```

## Citation 
Please cite our work if you use code from this repo:
```bibtex
@inproceedings{mcelfresh2023neural,
  title={When Do Neural Nets Outperform Boosted Trees on Tabular Data?}, 
  author={McElfresh, Duncan and Khandagale, Sujay and Valverde, Jonathan and Ramakrishnan, Ganesh and Prasad, Vishak and Goldblum, Micah and White, Colin}, 
  booktitle={Advances in Neural Information Processing Systems},
  year={2023}, 
} 
```
