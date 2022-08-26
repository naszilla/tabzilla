import gzip
import json
import time
from pathlib import Path

from models import all_models
from models.basemodel import BaseModel
from tabzilla_data_processing import process_data
from tabzilla_datasets import TabularDataset
from utils.scorer import BinScorer, ClassScorer, RegScorer
from utils.timer import Timer


def generate_filepath(name, extension):
    # generate filepath, of the format <name>_YYYYMMDD_HHMMDD.<extension>
    timestr = time.strftime("%Y%m%d_%H%M%S")
    return (name + "_%s." + extension) % timestr


def get_scorer(objective):
    if objective == "regression":
        return RegScorer()
    elif objective == "classification":
        return ClassScorer()
    elif objective == "binary":
        return BinScorer()
    else:
        raise NotImplementedError('No scorer for "' + objective + '" implemented')


class ExperimentResult:
    """
    container class for an experiment result.

    attributes:
    - dataset(TabularDataset): a dataset object
    - model(BaseModel): the model trained & evaluated on the dataset
    - timers(dict[Timer]): timers for training and evaluating model
    - scorers(dict): scorer objects for train, test, and val sets
    - predictions(dict): output of the model on the dataset. keys = "train", "test", "val"
    - probabilities(dict): probabilities of predicted class (only for classification problems)
    - ground_truth(dict): ground truth for each prediction, stored here just for convenience.
    - hparam_source(str): a string describing how the hyperparameters were generated
    - trial_number(int): trial number

    attributes "predictions", "probabilities", and "ground_truth" each have the same shape as the lists in dataset.split_indeces.
    """

    def __init__(
        self,
        dataset,
        model,
        timers,
        scorers,
        predictions,
        probabilities,
        ground_truth,
    ) -> None:
        self.dataset = dataset
        self.model = model
        self.timers = timers
        self.scorers = scorers
        self.predictions = predictions
        self.probabilities = probabilities
        self.ground_truth = ground_truth

        # we will set these after initialization
        self.hparam_source = None
        self.trial_number = None
        self.experiemnt_args = None

    def write(self, filepath, compress=False):
        """write all result properties to a new file. raise an exception if the file exists."""

        # create a dict with all output we want to store
        result_dict = {
            "dataset": self.dataset.get_metadata(),
            "model": self.model.get_metadata(),
            "experiemnt_args": self.experiment_args,
            "hparam_source": self.hparam_source,
            "trial_number": self.trial_number,
            "timers": {name: timer.save_times for name, timer in self.timers.items()},
            "scorers": {
                name: scorer.get_results() for name, scorer in self.scorers.items()
            },
            "splits": [
                {key: list(val.tolist()) for key, val in split.items()}
                for split in self.dataset.split_indeces
            ],
            "predictions": self.predictions,
            "probabilities": self.probabilities,
            "ground_truth": self.ground_truth,
        }

        write_dict_to_json(result_dict, filepath, compress=compress)


def cross_validation(model: BaseModel, dataset: TabularDataset) -> ExperimentResult:
    """
    takes a BaseModel and TabularDataset as input, and trains and evaluates the model using cross validation with all
    folds specified in the dataset property split_indeces

    returns an ExperimentResult object, which contains all metadata and results from the cross validation run, including:
    - evlaution objects for the validation and test sets
    - predictions and prediction probabilities for all data points in each fold.
    - runtimes for training and evaluation, for each fold
    """

    # Record some statistics and metrics
    # create a scorer & timer object for the train, val, and test sets
    scorers = {
        "train": get_scorer(dataset.target_type),
        "val": get_scorer(dataset.target_type),
        "test": get_scorer(dataset.target_type),
    }
    timers = {
        "train": Timer(),
        "val": Timer(),
        "test": Timer(),
        "train-eval": Timer(),
    }

    # store predictions and class probabilities. probs will be None for regression problems.
    # these have the same dimension as train_index, test_index, and val_index
    predictions = {
        "train": [],
        "val": [],
        "test": [],
    }
    probabilities = {
        "train": [],
        "val": [],
        "test": [],
    }
    ground_truth = {
        "train": [],
        "val": [],
        "test": [],
    }

    # iterate over all train/val/test splits in the dataset property split_indeces
    for i, split_dictionary in enumerate(dataset.split_indeces):

        train_index = split_dictionary["train"]
        val_index = split_dictionary["val"]
        test_index = split_dictionary["test"]

        # run pre-processing & split data
        processed_data = process_data(
            dataset,
            train_index,
            val_index,
            test_index,
            verbose=False,
            scale=False,
            one_hot_encode=False,
        )
        X_train, y_train = processed_data["data_train"]
        X_val, y_val = processed_data["data_val"]
        X_test, y_test = processed_data["data_test"]

        # Create a new unfitted version of the model
        curr_model = model.clone()

        # Train model
        timers["train"].start()
        # loss history can be saved if needed
        # TODO: check how X_test, y_test are used here. it appears that they are sometimes used for training... this would not be good.
        loss_history, val_loss_history = curr_model.fit(
            X_train,
            y_train,
            X_test,
            y_test,
        )
        timers["train"].end()

        # evaluate on train set
        timers["train-eval"].start()
        train_predictions, train_probs = curr_model.predict(X_train)
        timers["train-eval"].end()

        # evaluate on val set
        timers["val"].start()
        val_predictions, val_probs = curr_model.predict(X_val)
        timers["val"].end()

        # evaluate on test set
        timers["test"].start()
        test_predictions, test_probs = curr_model.predict(X_test)
        timers["test"].end()

        # evaluate on train, val, and test sets
        scorers["train"].eval(y_train, train_predictions, train_probs)
        scorers["val"].eval(y_val, val_predictions, val_probs)
        scorers["test"].eval(y_test, test_predictions, test_probs)

        # store predictions & ground truth

        # train
        predictions["train"].append(list(train_predictions))
        probabilities["train"].append(list(train_probs))
        ground_truth["train"].append(list(y_train))

        # val
        predictions["val"].append(list(val_predictions))
        probabilities["val"].append(list(val_probs))
        ground_truth["val"].append(list(y_val))

        # test
        predictions["test"].append(list(test_predictions))
        probabilities["test"].append(list(test_probs))
        ground_truth["test"].append(list(y_test))

    return ExperimentResult(
        dataset=dataset,
        model=model,
        timers=timers,
        scorers=scorers,
        predictions=predictions,
        probabilities=probabilities,
        ground_truth=ground_truth,
    )


def write_dict_to_json(x: dict, filepath: Path, compress=False):
    assert not filepath.is_file(), f"file already exists: {filepath}"
    assert filepath.parent.is_dir(), f"directory does not exist: {filepath.parent}"
    if not compress:
        with filepath.open("w", encoding="UTF-8") as f:
            json.dump(x, f)
    else:
        with gzip.open(str(filepath) + ".gz", "wb") as f:
            f.write(json.dumps(x).encode("UTF-8"))


import configargparse
import yaml

# the parsers below are based on the TabSurvey parsers in utils.py


def get_experiment_parser():
    """parser for experiment arguments"""

    experiment_parser = configargparse.ArgumentParser(
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
    )
    experiment_parser.add(
        "-experiment_config",
        required=True,
        is_config_file=True,
        help="config file for arg parser",
    )
    experiment_parser.add_argument(
        "--dataset_dir",
        required=True,
        type=str,
        help="directory containing pre-processed dataset.",
    )
    experiment_parser.add_argument(
        "--output_dir",
        required=True,
        type=str,
        help="directory where experiment results will be written.",
    )
    experiment_parser.add_argument(
        "--model_name",
        required=True,
        type=str,
        choices=all_models,
        help="name of the algorithm",
    )
    experiment_parser.add(
        "--use_gpu", action="store_true", help="Set to true if GPU is available"
    )
    experiment_parser.add(
        "--gpu_ids",
        type=int,
        action="append",
        help="IDs of the GPUs used when data_parallel is true",
    )
    experiment_parser.add(
        "--data_parallel",
        action="store_true",
        help="Distribute the training over multiple GPUs",
    )
    experiment_parser.add(
        "--n_random_trials",
        type=int,
        default=10,
        help="Number of trials of random hyperparameter search to run",
    )
    experiment_parser.add(
        "--hparam_seed",
        type=int,
        default=0,
        help="Random seed for generating random hyperparameters. passed to optuna RandomSampler.",
    )
    experiment_parser.add(
        "--n_opt_trials",
        type=int,
        default=10,
        help="Number of trials of hyperparameter optimization to run",
    )
    experiment_parser.add(
        "--batch_size", type=int, default=128, help="Batch size used for training"
    )
    experiment_parser.add(
        "--val_batch_size",
        type=int,
        default=128,
        help="Batch size used for training and testing",
    )
    experiment_parser.add(
        "--early_stopping_rounds",
        type=int,
        default=20,
        help="Number of rounds before early stopping applies.",
    )
    experiment_parser.add(
        "--epochs", type=int, default=1000, help="Max number of epochs to train."
    )
    experiment_parser.add(
        "--logging_period",
        type=int,
        default=100,
        help="Number of iteration after which validation is printed.",
    )

    return experiment_parser
