# experiment script for tabzilla
#
# this script runs an experiment specified by a config file

import argparse
import logging
import sys
import traceback
from collections import namedtuple
from pathlib import Path
from typing import NamedTuple

import optuna

optuna.logging.set_verbosity(optuna.logging.ERROR)

from models.basemodel import BaseModel
from tabzilla_alg_handler import ALL_MODELS, get_model
from tabzilla_datasets import TabularDataset
from tabzilla_utils import (
    ExperimentResult,
    cross_validation,
    get_experiment_parser,
    get_scorer,
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
        hparam_seed: int,
        random_parameters: bool,
        time_limit: int,
    ):
        #  BaseModel handle that will be initialized and trained
        self.model_handle = model_handle

        self.dataset = dataset
        self.experiment_args = experiment_args
        self.dataset.subset_random_seed = self.experiment_args.subset_random_seed
        # directory where results will be written
        self.output_path = Path(self.experiment_args.output_dir).resolve()

        # create the scorer, and get the direction of optimization from the scorer object
        sc_tmp = get_scorer(dataset.target_type)
        self.direction = sc_tmp.direction

        # if True, sample random hyperparameters. if False, sample using the optuna sampler object
        self.random_parameters = random_parameters

        # if random_parameters = True, then this is used to generate random hyperparameters
        self.hparam_seed = hparam_seed

        # time limit for any cross-validation cycle (seconds)
        self.time_limit = time_limit

    def __call__(self, trial):
        if self.random_parameters:
            # first trial is always default params. after that, sample using either random or optuna suggested hparams
            if trial.number == 0:
                trial_params = self.model_handle.default_parameters()
                hparam_source = "default"
            else:
                trial_params = self.model_handle.get_random_parameters(
                    trial.number + self.hparam_seed * 999
                )
                hparam_source = f"random_{trial.number}_s{self.hparam_seed}"

        else:
            trial_params = self.model_handle.define_trial_parameters(
                trial, None
            )  # the second arg was "args", and is not used by the function. so we will pass None instead
            hparam_source = f"sampler_{trial.number}"

        # Create model
        # pass a namespace "args" that contains all information needed to initialize the model.
        # this is a combination of dataset args and parameter search args
        # in TabSurvey, these were passed through an argparse args object
        arg_namespace = namedtuple(
            "args",
            [
                "model_name",
                "batch_size",
                "scale_numerical_features",
                "val_batch_size",
                "objective",
                "gpu_ids",
                "use_gpu",
                "epochs",
                "data_parallel",
                "early_stopping_rounds",
                "dataset",
                "cat_idx",
                "num_features",
                "subset_features",
                "subset_rows",
                "subset_features_method",
                "subset_rows_method",
                "cat_dims",
                "num_classes",
                "logging_period",
            ],
        )

        # if model class has epochs defined, use this number. otherwise, use the num epochs passed in args.
        if hasattr(self.model_handle, "default_epochs"):
            max_epochs = self.model_handle.default_epochs
        else:
            max_epochs = self.experiment_args.epochs

        args = arg_namespace(
            model_name=self.model_handle.__name__,
            batch_size=self.experiment_args.batch_size,
            val_batch_size=self.experiment_args.val_batch_size,
            scale_numerical_features=self.experiment_args.scale_numerical_features,
            epochs=max_epochs,
            gpu_ids=self.experiment_args.gpu_ids,
            use_gpu=self.experiment_args.use_gpu,
            data_parallel=self.experiment_args.data_parallel,
            early_stopping_rounds=self.experiment_args.early_stopping_rounds,
            logging_period=self.experiment_args.logging_period,
            objective=self.dataset.target_type,
            dataset=self.dataset.name,
            cat_idx=self.dataset.cat_idx,
            num_features=self.dataset.num_features,
            subset_features=self.experiment_args.subset_features,
            subset_rows=self.experiment_args.subset_rows,
            subset_features_method=self.experiment_args.subset_features_method,
            subset_rows_method=self.experiment_args.subset_rows_method,
            cat_dims=self.dataset.cat_dims,
            num_classes=self.dataset.num_classes,
        )

        # parameterized model
        model = self.model_handle(trial_params, args)

        # Cross validate the chosen hyperparameters
        try:
            result = cross_validation(
                model,
                self.dataset,
                self.time_limit,
                scaler=args.scale_numerical_features,
                args=args,
            )
            obj_val = result.scorers["val"].get_objective_result()
        except Exception as e:
            print(f"caught exception during cross-validation...")
            tb = traceback.format_exc()
            result = ExperimentResult(
                dataset=self.dataset,
                scaler=args.scale_numerical_features,
                model=model,
                timers={},
                scorers={},
                predictions=None,
                probabilities=None,
                ground_truth=None,
            )
            result.exception = tb
            obj_val = None
            print(tb)

        # add info about the hyperparams and trial number
        result.hparam_source = hparam_source
        result.trial_number = trial.number
        result.experiment_args = vars(self.experiment_args)

        # write results to file
        result_file_base = self.output_path.joinpath(
            f"{hparam_source}_trial{trial.number}"
        )
        result.write(
            result_file_base,
            write_predictions=self.experiment_args.write_predictions,
            compress=False,
        )

        return obj_val


def iteration_callback(study, trial):
    print(f"Trial {trial.number + 1} complete")


def main(experiment_args, model_name, dataset_dir):
    # read dataset from folder
    dataset = TabularDataset.read(Path(dataset_dir).resolve())

    model_handle = get_model(model_name)

    # create results directory if it doesn't already exist
    output_path = Path(experiment_args.output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

    if experiment_args.n_random_trials > 0:
        objective = TabZillaObjective(
            model_handle=model_handle,
            dataset=dataset,
            experiment_args=experiment_args,
            hparam_seed=experiment_args.hparam_seed,
            random_parameters=True,
            time_limit=experiment_args.trial_time_limit,
        )

        print(
            f"evaluating {experiment_args.n_random_trials} random hyperparameter samples..."
        )
        study = optuna.create_study(
            direction=objective.direction,
            study_name=None,
            storage=None,
            load_if_exists=False,
        )
        study.optimize(
            objective,
            n_trials=experiment_args.n_random_trials,
            timeout=experiment_args.experiment_time_limit,
            callbacks=[iteration_callback],
        )
        previous_trials = study.trials
    else:
        previous_trials = None

    if experiment_args.n_opt_trials:
        # TODO: this needs to be tested
        objective = TabZillaObjective(
            model_handle=model_handle,
            dataset=dataset,
            experiment_args=experiment_args,
            hparam_seed=experiment_args.hparam_seed,
            random_parameters=False,
            time_limit=experiment_args.trial_time_limit,
        )

        print(
            f"running {experiment_args.n_opt_trials} steps of hyperparameter optimization..."
        )
        study = optuna.create_study(
            direction=objective.direction,
            study_name=None,
            storage=None,
            load_if_exists=False,
        )
        # if random search was run, add these trials
        if previous_trials is not None:
            print(
                f"adding {experiment_args.n_random_trials} random trials to warm-start HPO"
            )
            study.add_trials(previous_trials)
        study.optimize(
            objective,
            n_trials=experiment_args.n_opt_trials,
            timeout=experiment_args.experiment_time_limit,
        )

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
