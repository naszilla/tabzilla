# experiment script for tabzilla
#
# this script takes three inputs:
# - a config file for search args (see tabzilla_utils.get_search_parser)
# - a config file for dataset args (see tabzilla_utils.get_dataset_parser)
# - the name of a model (algorithm) to train and evaluate the dataset on

########
# TODO:
# - change the output file name to something standardized, or take it as an arg

import argparse
import logging
import sys
from collections import namedtuple
from pathlib import Path

import optuna
from optuna.samplers import RandomSampler

from models import all_models, str2model
from tabzilla_datasets import TabularDataset
from tabzilla_utils import cross_validation, get_search_parser, get_scorer


class TabZillaObjective(object):
    """
    adapted from TabSurvey.train.Objective.
    this saves all metrics as user attributes for each trial.
    """

    def __init__(self, model_name, dataset, search_args, hparam_source):
        # Save the model that will be trained
        self.model_name = model_name

        self.dataset = dataset
        self.search_args = search_args

        # create the scorer, and get the direction of optimization from the scorer object
        sc_tmp = get_scorer(dataset.target_type)
        self.direction = sc_tmp.direction

        # this should be a string, which is set as a user attrivute
        self.hparam_source = hparam_source

    def __call__(self, trial):

        # TODO: we should limit the number of samples for algs with no hyperparams - right now, this is only LinearModel..
        # Define hyperparameters to optimize
        trial_params = self.model_name.define_trial_parameters(
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
            batch_size=self.search_args.batch_size,
            val_batch_size=self.search_args.val_batch_size,
            epochs=self.search_args.epochs,
            gpu_ids=self.search_args.gpu_ids,
            use_gpu=self.search_args.use_gpu,
            data_parallel=self.search_args.data_parallel,
            early_stopping_rounds=self.search_args.early_stopping_rounds,
            logging_period=self.search_args.logging_period,
            objective=self.dataset.target_type,
            dataset=self.dataset.name,
            cat_idx=self.dataset.cat_idx,
            num_features=self.dataset.num_features,
            cat_dims=self.dataset.cat_dims,
            num_classes=self.dataset.num_classes,
        )

        model = self.model_name(trial_params, args)

        # Cross validate the chosen hyperparameters
        sc_val, sc_test, train_timer, val_timer, test_timer = cross_validation(
            model, self.dataset
        )

        # save important attributes to trial_user_attr
        trial.set_user_attr("trial_params", trial_params)  # trial hyperparameters
        trial.set_user_attr("val_metrics", sc_val.get_results())
        trial.set_user_attr("test_metrics", sc_test.get_results())
        trial.set_user_attr("train_times", train_timer.save_times)
        trial.set_user_attr("val_times", val_timer.save_times)
        trial.set_user_attr("test_times", test_timer.save_times)
        trial.set_user_attr("hparam_source", self.hparam_source)

        return sc_val.get_objective_result()


def main(args, search_args):
    dataset = TabularDataset.read(Path(args.dataset_dir).resolve())

    model_handle = str2model(args.model_name)

    # all results will be written to the local sqlite database.
    # if this database exists, results will be added to it--this is due to the flag load_if_exists for optuna.create_study
    # NOTE: study_name should always be equivalent ot the database file name. this is necessary for reading the study database.
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = args.model_name + "_" + dataset.name
    storage_name = "sqlite:///{}.db".format(study_name)

    if search_args.n_random_trials > 0:

        objective = TabZillaObjective(model_handle, dataset, search_args, "random")

        print(
            f"evaluating {search_args.n_random_trials} random hyperparameter samples..."
        )
        study = optuna.create_study(
            direction=objective.direction,
            study_name=study_name,
            storage=storage_name,
            load_if_exists=True,
            sampler=RandomSampler(),
        )
        study.optimize(objective, n_trials=search_args.n_random_trials)
        previous_trials = study.trials
    else:
        previous_trials = None

    if search_args.n_opt_trials:

        objective = TabZillaObjective(
            model_handle, dataset, search_args, "optimization"
        )

        print(
            f"running {search_args.n_opt_trials} steps of hyperparameter optimization..."
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
                f"adding {search_args.n_random_trials} random trials to warm-start HPO"
            )
            study.add_trials(previous_trials)
        study.optimize(objective, n_trials=search_args.n_opt_trials)

    print(f"trials complete. results written to {storage_name}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="parser for tabzilla experiments")

    parser.add_argument(
        "--dataset_dir",
        required=True,
        type=str,
        help="dataset directory. dataset will be read using TabularDataset.read()",
    )
    parser.add_argument(
        "--search_config",
        required=True,
        type=str,
        help="config file for parameter search args",
    )
    parser.add_argument(
        "--model_name",
        required=True,
        type=str,
        choices=all_models,
        help="name of the algorithm",
    )

    args = parser.parse_args()
    print(f"ARGS: {args}")

    # now parse the dataset and search config files
    search_parser = get_search_parser()

    search_args = search_parser.parse_args(args="-search_config " + args.search_config)
    print(f"SEARCH ARGS: {search_args}")

    main(args, search_args)
