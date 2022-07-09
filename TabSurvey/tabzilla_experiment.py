# experiment script for tabzilla
#
# this script takes three inputs:
# - a config file for general args (see tabzilla_utils.get_general_parser)
# - a config file for dataset args (see tabzilla_utils.get_dataset_parser)
# - the name of a model (algorithm) to train and evaluate the dataset on

########
# TODO:
# - change the output file name to something standardized, or take it as an arg
# - do randomized hyperparameter search


import logging
import sys

import optuna

import argparse 
from argparse import Namespace

from models import str2model, all_models
from utils.load_data import load_data
from utils.parser import get_parser, get_given_parameters_parser
from utils.io_utils import save_results_to_file, save_hyperparameters_to_file, save_loss_to_file
from utils.scorer import get_scorer
from tabzilla_utils import trial_to_dict, get_general_parser, get_dataset_parser

from train import cross_validation

class TabZillaObjective(object):
    """adapted from TabSurvey.train.Objective. this saves all metrics as user attributes for each trial."""

    def __init__(self, args, model_name, X, y):
        # Save the model that will be trained
        self.model_name = model_name

        # Save the trainings data
        self.X = X
        self.y = y

        self.args = args

    def __call__(self, trial):
        # Define hyperparameters to optimize
        trial_params = self.model_name.define_trial_parameters(trial, self.args)

        # Create model
        model = self.model_name(trial_params, self.args)

        # Cross validate the chosen hyperparameters
        sc, time = cross_validation(model, self.X, self.y, self.args)

        save_hyperparameters_to_file(self.args, trial_params, sc.get_results(), time)

        # save important attributes to trial user attributes
        trial.set_user_attr("trial_params", trial_params)  # trial hyperparameters
        trial.set_user_attr("metrics", sc.get_results())  # dict of performance metrics defined in scorer.py
        trial.set_user_attr("avg_train_time", time[0])
        trial.set_user_attr("avg_test_time", time[1])

        return sc.get_objective_result()


def main(args):
    print("Start hyperparameter optimization")
    X, y = load_data(args)

    model_name = str2model(args.model_name)

    sc = get_scorer(args)

    # all results will be written to the local sqlite database. 
    # if this database exists, results will be added to it--this is due to the flag load_if_exists for optuna.create_study
    # NOTE: study_name should always be equivalent ot the database file name. this is necessary for reading the study database.
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = args.model_name + "_" + args.dataset
    storage_name = "sqlite:///{}.db".format(study_name)

    study = optuna.create_study(
        direction=args.direction,
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
    )
    study.optimize(TabZillaObjective(args, model_name, X, y), n_trials=args.n_trials)
    print(f"trials complete. results written to {storage_name}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='parser for tabzilla experiments')

    parser.add_argument('--dataset_config', required=True, type=str, help='config file for dataset args')
    parser.add_argument('--general_config', required=True, type=str, help='config file for general args')
    parser.add_argument('--model_name', required=True, type=str, choices=all_models, help="name of the algorithm")
    
    args = parser.parse_args()
    print(f"ARGS: {args}" )

    # now parse the dataset and general config files
    dataset_parser = get_dataset_parser()
    general_parser = get_general_parser()

    dataset_args = dataset_parser.parse_args(args="-data_config " + args.dataset_config)
    print(f"DATASET ARGS: {dataset_args}")

    general_args = general_parser.parse_args(args="-gen_config " + args.general_config)
    print(f"GENERAL ARGS: {general_args}")

    # combine all arge
    combined_args = Namespace(**vars(args), **vars(dataset_args), **vars(general_args))

    print(f"COMBINED ARGS: {combined_args}")

    main(combined_args)
