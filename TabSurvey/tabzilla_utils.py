import json
from pathlib import Path

from sklearn.model_selection import KFold, StratifiedKFold
from tabzilla_data_processing import process_data
from utils.scorer import BinScorer, ClassScorer, RegScorer

from optuna.trial import FrozenTrial

from models.basemodel import BaseModel
from tabzilla_datasets import TabularDataset
from utils.timer import Timer


def get_scorer(objective):
    if objective == "regression":
        return RegScorer()
    elif objective == "classification":
        return ClassScorer()
    elif objective == "binary":
        return BinScorer()
    else:
        raise NotImplementedError('No scorer for "' + args.objective + '" implemented')


def cross_validation(
    model: BaseModel,
    dataset: TabularDataset,
    save_model=False,
    seed=0,
    num_splits=5,
    shuffle=True,
):
    """
    adapted from TabSurvey.train.cross_validation.

    takes a BaseModel and TabularDataset as input, and trains and evaluates the model using k-fold cross
    validation on the dataset. for regression, use k-fold. for classification & binary, use stratified k-fold"""

    # Record some statistics and metrics
    sc = get_scorer(dataset.target_type)
    train_timer = Timer()
    test_timer = Timer()

    if dataset.target_type == "regression":
        kf = KFold(n_splits=num_splits, shuffle=shuffle, random_state=seed)
    elif dataset.target_type == "classification" or dataset.target_type == "binary":
        kf = StratifiedKFold(n_splits=num_splits, shuffle=shuffle, random_state=seed)
    else:
        raise NotImplementedError(
            "Objective" + dataset.target_type + "is not yet implemented."
        )

    for i, (train_index, test_index) in enumerate(kf.split(dataset.X, dataset.y)):

        # run pre-processing & split data
        # TODO: maybe make a pre-processing object for passing all of these args.. or, attach this to the TabularDataset object.
        # TODO: pass these processing args from the experiment function, somehow...
        processed_data = process_data(
            dataset,
            train_index,
            test_index,
            verbose=False,
            scale=False,
            one_hot_encode=False,
        )
        X_train, y_train = processed_data["data_train"]
        X_test, y_test = processed_data["data_test"]

        # Create a new unfitted version of the model
        curr_model = model.clone()

        # Train model
        train_timer.start()
        loss_history, val_loss_history = curr_model.fit(
            X_train, y_train, X_test, y_test
        )
        train_timer.end()

        # Test model
        test_timer.start()
        curr_model.predict(X_test)
        test_timer.end()

        # Save model weights and the truth/prediction pairs for traceability
        curr_model.save_model_and_predictions(y_test, i)

        # TODO: remove if not needed
        # if save_model:
        #     save_loss_to_file(args, loss_history, "loss", extension=i)
        #     save_loss_to_file(args, val_loss_history, "val_loss", extension=i)

        # Compute scores on the output
        sc.eval(y_test, curr_model.predictions, curr_model.prediction_probabilities)

        # print(sc.get_results())

    # Best run is saved to file
    # TODO: remove if not needed
    # if save_model:
    #     print("Results:", sc.get_results())
    #     print("Train time:", train_timer.get_average_time())
    #     print("Inference time:", test_timer.get_average_time())

    #     # Save the all statistics to a file
    #     save_results_to_file(
    #         args,
    #         sc.get_results(),
    #         train_timer.get_average_time(),
    #         test_timer.get_average_time(),
    #         model.params,
    #     )

    # print("Finished cross validation")
    return sc, (train_timer.get_average_time(), test_timer.get_average_time())


def write_dict_to_json(x: dict, filepath: Path):
    assert not filepath.is_file(), f"file already exists: {filepath}"
    assert filepath.parent.is_dir(), f"directory does not exist: {filepath.parent}"
    with filepath.open("w", encoding="UTF-8") as f:
        json.dump(x, f)


def trial_to_dict(trial):
    """return a dict representation of an optuna FrozenTrial"""
    assert isinstance(
        trial, FrozenTrial
    ), f"trial must be of type optuna.trial.FrozenTrial. this object has type {type(trial)}"

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

    dataset_parser = configargparse.ArgumentParser(
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
    )

    dataset_parser.add(
        "-data_config",
        required=True,
        is_config_file=True,
        help="optional config file for dataset parser",
    )

    dataset_parser.add(
        "--dataset", required=True, help="Name of the dataset that will be used"
    )
    dataset_parser.add(
        "--objective",
        required=True,
        type=str,
        default="regression",
        choices=["regression", "classification", "binary"],
        help="Set the type of the task",
    )
    dataset_parser.add(
        "--direction",
        type=str,
        default="minimize",
        choices=["minimize", "maximize"],
        help="Direction of optimization.",
    )

    dataset_parser.add(
        "--num_features",
        type=int,
        required=True,
        help="Set the total number of features.",
    )
    dataset_parser.add(
        "--num_classes",
        type=int,
        default=1,
        help="Set the number of classes in a classification task.",
    )
    dataset_parser.add(
        "--cat_idx",
        type=int,
        action="append",
        help="Indices of the categorical features",
    )
    dataset_parser.add(
        "--cat_dims",
        type=int,
        action="append",
        help="Cardinality of the categorical features (is set "
        "automatically, when the load_data function is used.",
    )

    dataset_parser.add("--scale", action="store_true", help="Normalize input data.")
    dataset_parser.add(
        "--target_encode",
        action="store_true",
        help="Encode the targets that they start at 0. (0, 1, 2,...)",
    )
    dataset_parser.add(
        "--one_hot_encode",
        action="store_true",
        help="OneHotEncode the categorical features",
    )

    return dataset_parser


def get_search_parser():
    """parser for parameter search args only"""

    search_parser = configargparse.ArgumentParser(
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
    )

    search_parser.add(
        "-search_config",
        required=True,
        is_config_file=True,
        help="optional config file for search parser",
    )

    search_parser.add(
        "--use_gpu", action="store_true", help="Set to true if GPU is available"
    )
    search_parser.add(
        "--gpu_ids",
        type=int,
        action="append",
        help="IDs of the GPUs used when data_parallel is true",
    )
    search_parser.add(
        "--data_parallel",
        action="store_true",
        help="Distribute the training over multiple GPUs",
    )

    search_parser.add(
        "--optimize_hyperparameters",
        action="store_true",
        help="Search for the best hyperparameters",
    )
    search_parser.add(
        "--n_trials",
        type=int,
        default=100,
        help="Number of trials for the hyperparameter optimization",
    )

    search_parser.add(
        "--num_splits",
        type=int,
        default=5,
        help="Number of splits done for cross validation",
    )
    search_parser.add(
        "--shuffle", action="store_true", help="Shuffle data during cross-validation"
    )
    search_parser.add(
        "--seed", type=int, default=123, help="Seed for KFold initialization."
    )

    search_parser.add(
        "--batch_size", type=int, default=128, help="Batch size used for training"
    )
    search_parser.add(
        "--val_batch_size",
        type=int,
        default=128,
        help="Batch size used for training and testing",
    )
    search_parser.add(
        "--early_stopping_rounds",
        type=int,
        default=20,
        help="Number of rounds before early stopping applies.",
    )
    search_parser.add(
        "--epochs", type=int, default=1000, help="Max number of epochs to train."
    )
    search_parser.add(
        "--logging_period",
        type=int,
        default=100,
        help="Number of iteration after which validation is printed.",
    )

    return search_parser
