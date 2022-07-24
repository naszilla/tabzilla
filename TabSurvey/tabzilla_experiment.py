# experiment script for tabzilla
#
# this script takes three inputs:
# - a config file for search args (see tabzilla_utils.get_search_parser)
# - a config file for dataset args (see tabzilla_utils.get_dataset_parser)
# - the name of a model (algorithm) to train and evaluate the dataset on

########
# TODO:
# - change the output file name to something standardized, or take it as an arg
# - do randomized hyperparameter search


import argparse
import logging
import sys
from collections import namedtuple

import optuna

from models import all_models, str2model
from tabzilla_datasets import TabularDataset
from tabzilla_utils import 
from tabzilla_utils import cross_validation, get_search_parser


class TabZillaObjective(object):
    """
    adapted from TabSurvey.train.Objective.
    this saves all metrics as user attributes for each trial.
    """

    def __init__(self, model_name, dataset, search_args):
        # Save the model that will be trained
        self.model_name = model_name

        self.dataset = dataset
        self.search_args = search_args

    def __call__(self, trial):
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
        sc, time = cross_validation(
            model, self.dataset, save_model=False, seed=0, num_splits=5, shuffle=True
        )

        # save important attributes to trial_user_attr
        trial.set_user_attr("trial_params", trial_params)  # trial hyperparameters
        trial.set_user_attr(
            "metrics", sc.get_results()
        )  # dict of performance metrics defined in scorer.py
        trial.set_user_attr("avg_train_time", time[0])
        trial.set_user_attr("avg_test_time", time[1])

        return sc.get_objective_result()


def main(args, search_args):
    print("Start hyperparameter optimization")
    dataset = TabularDataset.read(args.dataset_dir)

    model_handle = str2model(args.model_name)

    # all results will be written to the local sqlite database.
    # if this database exists, results will be added to it--this is due to the flag load_if_exists for optuna.create_study
    # NOTE: study_name should always be equivalent ot the database file name. this is necessary for reading the study database.
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = args.model_name + "_" + dataset.name
    storage_name = "sqlite:///{}.db".format(study_name)

    study = optuna.create_study(
        direction=dataset.direction,
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
    )
    study.optimize(
        TabZillaObjective(model_handle, dataset, search_args),
        n_trials=search_args.n_trials,
    )
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
