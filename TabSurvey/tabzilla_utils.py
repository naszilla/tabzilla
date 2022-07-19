from pathlib import Path
import json
from optuna.trial import FrozenTrial

def write_dict_to_json(x: dict, filepath: Path):
    assert not filepath.is_file(), f"file already exists: {filepath}"
    assert filepath.parent.is_dir(), f"directory does not exist: {filepath.parent}"
    with filepath.open("w", encoding="UTF-8") as f: 
        json.dump(x, f)

def trial_to_dict(trial):
    """return a dict representation of an optuna FrozenTrial"""
    assert isinstance(trial, FrozenTrial), f"trial must be of type optuna.trial.FrozenTrial. this object has type {type(trial)}"
    
    # get all user_metrics
    trial_dict = trial.user_attrs.copy()

    # add trial number
    trial_dict["trial_number"] = trial.number
    
    # add trial number
    trial_dict["trial_params_obj"] = trial.params
    
    # add system attributes
    trial_dict["system_attributes"] = trial.system_attrs

    return trial_dict

def write_trial_to_json(trial, filepath: Path):
    """write the dict representation of an optuna trial to file"""
    write_dict_to_json(trial_to_dict(trial), filepath)



import configargparse
import yaml


# the parsers below are based on the TabSurvey parsers in utils.py

def get_dataset_parser():
    """parser for dataset args only"""

    dataset_parser = configargparse.ArgumentParser(config_file_parser_class=configargparse.YAMLConfigFileParser,
                                           formatter_class=configargparse.ArgumentDefaultsHelpFormatter)

    dataset_parser.add("-data_config", required=True, is_config_file=True, help="optional config file for dataset parser")

    dataset_parser.add('--dataset', required=True, help="Name of the dataset that will be used")
    dataset_parser.add('--objective', required=True, type=str, default="regression", choices=["regression", "classification",
                                                                                      "binary"],
               help="Set the type of the task")
    dataset_parser.add('--direction', type=str, default="minimize", choices=['minimize', 'maximize'],
               help="Direction of optimization.")

    dataset_parser.add('--num_features', type=int, required=True, help="Set the total number of features.")
    dataset_parser.add('--num_classes', type=int, default=1, help="Set the number of classes in a classification task.")
    dataset_parser.add('--cat_idx', type=int, action="append", help="Indices of the categorical features")
    dataset_parser.add('--cat_dims', type=int, action="append", help="Cardinality of the categorical features (is set "
                                                             "automatically, when the load_data function is used.")

    dataset_parser.add('--scale', action="store_true", help="Normalize input data.")
    dataset_parser.add('--target_encode', action="store_true", help="Encode the targets that they start at 0. (0, 1, 2,...)")
    dataset_parser.add('--one_hot_encode', action="store_true", help="OneHotEncode the categorical features")

    return dataset_parser


def get_general_parser():
    """parser for general args only"""

    general_parser = configargparse.ArgumentParser(config_file_parser_class=configargparse.YAMLConfigFileParser,
                                           formatter_class=configargparse.ArgumentDefaultsHelpFormatter)
    
    general_parser.add("-gen_config", required=True, is_config_file=True, help="optional config file for general parser")

    general_parser.add('--use_gpu', action="store_true", help="Set to true if GPU is available")
    general_parser.add('--gpu_ids', type=int, action="append", help="IDs of the GPUs used when data_parallel is true")
    general_parser.add('--data_parallel', action="store_true", help="Distribute the training over multiple GPUs")

    general_parser.add('--optimize_hyperparameters', action="store_true",
               help="Search for the best hyperparameters")
    general_parser.add('--n_trials', type=int, default=100, help="Number of trials for the hyperparameter optimization")

    general_parser.add('--num_splits', type=int, default=5, help="Number of splits done for cross validation")
    general_parser.add('--shuffle', action="store_true", help="Shuffle data during cross-validation")
    general_parser.add('--seed', type=int, default=123, help="Seed for KFold initialization.")

    general_parser.add('--batch_size', type=int, default=128, help="Batch size used for training")
    general_parser.add('--val_batch_size', type=int, default=128, help="Batch size used for training and testing")
    general_parser.add('--early_stopping_rounds', type=int, default=20, help="Number of rounds before early stopping applies.")
    general_parser.add('--epochs', type=int, default=1000, help="Max number of epochs to train.")
    general_parser.add('--logging_period', type=int, default=100, help="Number of iteration after which validation is printed.")

    return general_parser
