# experiment script for tabzilla
#
# this script runs an experiment specified by a config file

import argparse
import logging
import sys
from collections import namedtuple
from pathlib import Path
from typing import NamedTuple

import optuna
from optuna.samplers import RandomSampler

from models.basemodel import BaseModel
from tabzilla_alg_handler import ALL_MODELS, get_model
from tabzilla_datasets import TabularDataset
from tabzilla_utils import (
    cross_validation,
    generate_filepath,
    get_experiment_parser,
    get_scorer,
    make_archive,
)


class TabZillaObjective(object):
    """
    adapted from TabSurvey.train.Objective.
    this saves output from each trial.
    """

    def __init__(
        self,
        model_handle: BaseModel,
        dataset: TabularDataset,
        experiment_args: NamedTuple,
        hparam_source: str,
    ):
        #  BaseModel handle that will be initialized and trained
        self.model_handle = model_handle

        self.dataset = dataset
        self.experiment_args = experiment_args

        # directory where results will be written
        self.output_path = Path(self.experiment_args.output_dir).resolve()

        # create the scorer, and get the direction of optimization from the scorer object
        sc_tmp = get_scorer(dataset.target_type)
        self.direction = sc_tmp.direction

        # this should be a string that indicates the source of the hyperparameters
        self.hparam_source = hparam_source

        # to keep track of the number of evaluations, separate from the trial number
        self.counter = 0

    def __call__(self, trial):

        # TODO: we should limit the number of samples for algs with no hyperparams - right now, this is only LinearModel..
        # Define hyperparameters to optimize
        trial_params = self.model_handle.define_trial_parameters(
            trial, None
        )  # the second arg was "args", and is not used by the function. so we will pass None instead

        # Create model
        # pass a namespace "args" that contains all information needed to initialize the model.
        # this is a combination of dataset args and parameter search args
        # in TabSurvey, these were passed through an argparse args object
        arg_namespace = namedtuple(
            "args",
            [
                "batch_size",
                "val_batch_size",
                "objective",
                "epochs",
                "gpu_ids",
                "use_gpu",
                "data_parallel",
                "early_stopping_rounds",
                "dataset",
                "cat_idx",
                "num_features",
                "cat_dims",
                "num_classes",
                "logging_period",
            ],
        )

        args = arg_namespace(
            batch_size=self.experiment_args.batch_size,
            val_batch_size=self.experiment_args.val_batch_size,
            epochs=self.experiment_args.epochs,
            gpu_ids=self.experiment_args.gpu_ids,
            use_gpu=self.experiment_args.use_gpu,
            data_parallel=self.experiment_args.data_parallel,
            early_stopping_rounds=self.experiment_args.early_stopping_rounds,
            logging_period=self.experiment_args.logging_period,
            objective=self.dataset.target_type,
            dataset=self.dataset.name,
            cat_idx=self.dataset.cat_idx,
            num_features=self.dataset.num_features,
            cat_dims=self.dataset.cat_dims,
            num_classes=self.dataset.num_classes,
        )

        # parameterized model
        model = self.model_handle(trial_params, args)

        # Cross validate the chosen hyperparameters
        result = cross_validation(model, self.dataset)

        # add info about the hyperparams and trial number
        result.hparam_source = self.hparam_source
        result.trial_number = self.counter
        result.experiment_args = vars(self.experiment_args)

        # write result to file
        result_file = self.output_path.joinpath(
            generate_filepath(f"{self.hparam_source}_trial{self.counter}", "json")
        )
        result.write(result_file, compress=False)

        self.counter += 1

        return result.scorers["val"].get_objective_result()


def main(experiment_args, model_name, dataset_dir):

    # read dataset from folder
    dataset = TabularDataset.read(Path(dataset_dir).resolve())

    model_handle = get_model(model_name)

    # create results directory if it doesn't already exist
    output_path = Path(experiment_args.output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    # all results will be written to the local sqlite database.
    # if this database exists, results will be added to it--this is due to the flag load_if_exists for optuna.create_study
    # NOTE: study_name should always be equivalent ot the database file name. this is necessary for reading the study database.
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = model_name + "_" + dataset.name
    storage_name = "sqlite:///{}.db".format(study_name)

    if experiment_args.n_random_trials > 0:
        objective = TabZillaObjective(
            model_handle=model_handle,
            dataset=dataset,
            experiment_args=experiment_args,
            hparam_source=f"random_seed{experiment_args.hparam_seed}",
        )

        print(
            f"evaluating {experiment_args.n_random_trials} random hyperparameter samples..."
        )
        study = optuna.create_study(
            direction=objective.direction,
            study_name=study_name,
            storage=storage_name,
            load_if_exists=True,
            sampler=RandomSampler(experiment_args.hparam_seed),
        )
        study.optimize(objective, n_trials=experiment_args.n_random_trials)
        previous_trials = study.trials
    else:
        previous_trials = None

    if experiment_args.n_opt_trials:

        objective = TabZillaObjective(
            model_handle=model_handle,
            dataset=dataset,
            experiment_args=experiment_args,
            hparam_source="optimization",
        )

        print(
            f"running {experiment_args.n_opt_trials} steps of hyperparameter optimization..."
        )
        study = optuna.create_study(
            direction=objective.direction,
            study_name=study_name,
            storage=storage_name,
            load_if_exists=True,
        )
        # if random search was run, add these trials
        if previous_trials is not None:
            print(
                f"adding {experiment_args.n_random_trials} random trials to warm-start HPO"
            )
            study.add_trials(previous_trials)
        study.optimize(objective, n_trials=experiment_args.n_opt_trials)

    print(f"trials complete. results written to {output_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="parser for tabzilla experiments")

    parser.add_argument(
        "--experiment_config",
        required=True,
        type=str,
        help="config file for parameter experiment args",
    )

    parser.add_argument(
        "--dataset_dir",
        required=True,
        type=str,
        help="directory containing pre-processed dataset.",
    )
    parser.add_argument(
        "--model_name",
        required=True,
        type=str,
        choices=ALL_MODELS,
        help="name of the algorithm",
    )
    args = parser.parse_args()
    print(f"ARGS: {args}")

    # now parse the dataset and search config files
    experiment_parser = get_experiment_parser()

    experiment_args = experiment_parser.parse_args(
        args="-experiment_config " + args.experiment_config
    )
    print(f"EXPERIMENT ARGS: {experiment_args}")

    main(experiment_args, args.model_name, args.dataset_dir)
