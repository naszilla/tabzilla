import functools

import numpy as np
import openml
import pandas as pd
from tabzilla_preprocessor_utils import cv_n_folds, dataset_preprocessor

preprocessor_dict = {}

easy_import_task_file = "openml_easy_import_list.txt"  # Datasets identified just by their ID can be easily imported from here

debug_mode = False

openml_tasks = [
    {
        "openml_task_id": 3021,
        "drop_features": ["TBG"],
    },
    # These datasets have been added to openml_easy_import_list.txt, but are provided here as a blueprint
    # for addition of other datasets
    # {
    #     "openml_task_id": 361089,
    #     # "target_type": "regression", # Does not need to be explicitly specified, but can be
    # },
    # {
    #     "openml_task_id": 7592,
    #     # "target_type": "binary", # Does not need to be explicitly specified, but can be
    #     # "force_cat_features": ["workclass", "education"], # Example (these are not needed in this case)
    #     # "force_num_features": ["fnlwgt", "education-num"], # Example (these are not needed in this case)
    # },
]

with open(easy_import_task_file, "r") as f:
    for line in f:
        processed = line.strip()
        # Ignore empty lines
        if not processed:
            continue
        task_id = int(processed)
        openml_tasks.append({"openml_task_id": task_id})


# Based on: https://github.com/releaunifreiburg/WellTunedSimpleNets/blob/main/utilities.py
def preprocess_openml(
    openml_task_id,
    target_type=None,
    force_cat_features=None,
    force_num_features=None,
    drop_features=None,
):
    if force_num_features is None:
        force_num_features = []
    if force_cat_features is None:
        force_cat_features = []
    if drop_features is None:
        drop_features = []

    task = openml.tasks.get_task(task_id=openml_task_id)
    n_repeats, n_folds, n_samples = task.get_split_dimensions()
    if n_repeats != 1 or n_folds != cv_n_folds or n_samples != 1:
        raise NotImplementedError(
            f"Re-splitting required for split dimensions {n_repeats}, {n_folds}, {n_samples}."
        )

    # Extract splits
    split_indeces = []
    for split_idx in range(cv_n_folds):
        train_indices, test_indices = task.get_train_test_split_indices(
            repeat=0, fold=split_idx
        )
        split_indeces.append({"train": train_indices, "test": test_indices, "val": []})
    # Build validation set by using n+1 th test set.
    for split_idx in range(cv_n_folds):
        split_indeces[split_idx]["val"] = split_indeces[(split_idx + 1) % cv_n_folds][
            "test"
        ].copy()
        split_indeces[split_idx]["train"] = np.setdiff1d(
            split_indeces[split_idx]["train"],
            split_indeces[split_idx]["val"],
            assume_unique=True,
        )

    if debug_mode:
        # Sanity check
        for split_idx in range(cv_n_folds):
            print(
                "Dimensions: {}, {}, {}".format(
                    len(split_indeces[split_idx]["train"]),
                    len(split_indeces[split_idx]["val"]),
                    len(split_indeces[split_idx]["test"]),
                )
            )
            assert (
                len(
                    np.intersect1d(
                        split_indeces[split_idx]["train"],
                        split_indeces[split_idx]["test"],
                    )
                )
                == 0
            )
            assert (
                len(
                    np.intersect1d(
                        split_indeces[split_idx]["train"],
                        split_indeces[split_idx]["val"],
                    )
                )
                == 0
            )
            assert (
                len(
                    np.intersect1d(
                        split_indeces[split_idx]["test"],
                        split_indeces[split_idx]["val"],
                    )
                )
                == 0
            )
            for split_jdx in range(split_idx + 1, cv_n_folds):
                intersect = np.intersect1d(
                    split_indeces[split_idx]["test"], split_indeces[split_jdx]["test"]
                )
                assert len(intersect) == 0

    dataset = task.get_dataset()
    X, y, categorical_indicator, col_names = dataset.get_data(
        dataset_format="dataframe",
        target=task.target_name,
    )

    # Patch categorical columns and numerical columns in case of any mislabeling
    for feature_name in force_cat_features:
        categorical_indicator[col_names.index(feature_name)] = True
    for feature_name in force_num_features:
        categorical_indicator[col_names.index(feature_name)] = False

    # Drop features
    if drop_features:
        drop_indices = sorted([col_names.index(col) for col in drop_features])
        X.drop(columns=drop_features, inplace=True)
        categorical_indicator_old, categorical_indicator = categorical_indicator, []
        col_names_old, col_names = col_names, []
        for idx, (col_name, is_categorical) in enumerate(
            zip(col_names_old, categorical_indicator_old)
        ):
            if idx in drop_indices:
                continue
            col_names.append(col_name)
            categorical_indicator.append(is_categorical)

    # Run checks
    openml_data_dict = {
        "X": X,
        "y": y,
        "categorical_indicator": categorical_indicator,
        "col_names": col_names,
    }
    errors = inspect_openml_task(
        openml_task_id, openml_data_dict=openml_data_dict, verbose=False, debug=False
    )
    if errors:
        print("Dataset checks failed:")
        for error in errors:
            print(error)
        raise RuntimeError("One or more dataset checks failed!")

    cat_idx = [idx for idx, indicator in enumerate(categorical_indicator) if indicator]

    # Infer task type if not provided
    if target_type is None:
        if task.task_type == "Supervised Regression":
            target_type = "regression"
        elif task.task_type == "Supervised Classification":
            # n_unique_labels = len(task.class_labels)
            n_unique_labels = len(np.unique(y))
            metadata_num_classes = dataset.qualities["NumberOfClasses"]
            if (
                not np.isnan(metadata_num_classes)
                and metadata_num_classes != n_unique_labels
            ):
                raise RuntimeError(
                    "Inconsistent metadata. Cannot automatically infer task type."
                )
            if n_unique_labels == 2:
                target_type = "binary"
            elif n_unique_labels > 2:
                target_type = "classification"
            else:
                raise RuntimeError(
                    f"Unexpected number of class labels: {n_unique_labels}"
                )
        else:
            raise RuntimeError(f"Unsupported task type: {task.task_type}")

    # Get num_classes
    if target_type == "regression":
        num_classes = 1
    elif target_type == "binary":
        num_classes = 1
    elif target_type == "classification":
        num_classes = len(y.unique())
    else:
        raise RuntimeError(f"Unrecognized target_type: {target_type}")

    return {
        "X": X.values,
        "y": np.array(y),
        "cat_idx": cat_idx,
        "target_type": target_type,
        "num_classes": num_classes,
        "split_indeces": split_indeces,
        "split_source": "openml",
    }


def inspect_openml_task(
    openml_task_id,
    openml_data_dict=None,
    accept_nans=True,
    verbose=True,
    debug=True,
    exploratory_mode=False,
):
    """
    Use this function to inspect an OpenML task. The checks include making sure that the task is of the correct type
    (supervised classification or regression), checking missing values, cross validation, and validity of categorical
    column labels.

    Args:
        openml_task_id: The OpenML task ID, specified as an integ
        openml_data_dict (optional): specify a dictionary with entries for X, y, categorical_indicator, col_names.
            This is not needed for manual checking of datasets and is only used by the pre-processor function.
        debug: set to False if you do not want to invoke the debugger if the dataset does not pass any of the checks.
        accept_nans: set to False if you want the dataset to fail if there are any NaN features.
        verbose: set to False to suppress the console output.
        exploratory_mode: set to True if you want to invoke the debugger at the end of execution to explore the dataset,
            regardless of whether the checks were passed

    Returns:
        err_messages: list of strings of error messages. Empty if no errors were returned.
    """
    if verbose:
        print(f"TASK ID: {openml_task_id}")
    task = openml.tasks.get_task(task_id=openml_task_id)

    if openml_data_dict is None:
        dataset = task.get_dataset()
        X, y, categorical_indicator, col_names = dataset.get_data(
            dataset_format="dataframe",
            target=task.target_name,
        )
    else:
        X, y, categorical_indicator, col_names = (
            openml_data_dict["X"],
            openml_data_dict["y"],
            openml_data_dict["categorical_indicator"],
            openml_data_dict["col_names"],
        )

    err_messages = []

    def check_true(condition, message):
        if not condition:
            err_messages.append(message)

    # Task type check
    check_true(
        (task.task_type == "Supervised Regression")
        or (task.task_type == "Supervised Classification"),
        f"Wrong task type: {task.task_type}",
    )

    # Missing values checks
    num_na = X.isna().sum().sum()
    if num_na != 0:
        if verbose:
            print("Warning: {} missing values.".format(num_na))
        check_true(accept_nans, "NaNs not accepted.")

    null_cols = X.isnull().all()
    null_cols = null_cols[null_cols].index.to_list()
    check_true(not null_cols, f"Found full null columns: {null_cols}")
    check_true(y.isna().sum() == 0, "Missing labels.")

    # Cross validation checks
    estimation = task.estimation_procedure
    check_true(estimation["type"] == "crossvalidation", "Not cross validation")
    check_true(
        estimation["parameters"]["number_repeats"] == "1", "Wrong number_repeats"
    )
    check_true(
        estimation["parameters"]["number_folds"] == str(cv_n_folds),
        "Wrong number of folds",
    )

    # Categorical indicator checks
    cat_mask = np.array(categorical_indicator)
    # Confirm all columns labeled as categorical are categorical
    check_true(
        len(X.loc[:, cat_mask].select_dtypes("category").columns) == cat_mask.sum(),
        "Mislabeled numerical columns",
    )
    # Confirm rest of the columns are all numerical
    check_true(
        len(X.loc[:, ~cat_mask].select_dtypes("number").columns) == (~cat_mask).sum(),
        "Mislabeled categorical columns",
    )
    num_types = list(X.loc[:, ~cat_mask].dtypes.unique())
    cat_types = list(X.loc[:, cat_mask].dtypes.unique())

    if not err_messages:
        if verbose:
            print("Tests passed!")
        if exploratory_mode:
            breakpoint()
    else:
        if verbose:
            print("Errors found:")
            for message in err_messages:
                print(message)
        if debug or exploratory_mode:
            breakpoint()

    return err_messages


def get_openml_task_metadata(save=False):
    """
    Gets a dataframe with all OpenML supervised classification and regression tasks, along with a column that indicates
    whether the task has been imported into the repository.
    Returns:
    task_df: the dataframe.
    """
    task_types = [
        openml.tasks.task.TaskType.SUPERVISED_CLASSIFICATION,
        openml.tasks.task.TaskType.SUPERVISED_REGRESSION,
    ]
    task_df = [
        openml.tasks.list_tasks(task_type, output_format="dataframe")
        for task_type in task_types
    ]
    task_df = pd.concat(task_df)
    task_df.set_index("tid", inplace=True)

    implemented_tasks = [kwargs["openml_task_id"] for kwargs in openml_tasks]

    task_df["in_repo"] = False
    task_df.loc[implemented_tasks, "in_repo"] = True
    in_repo = task_df.pop("in_repo")
    task_df.insert(0, "in_repo", in_repo)

    if save:
        task_df.to_csv("openml_task_metadata.csv")

    return task_df


def check_tasks_from_suite(suite_id):
    """
    Goes through the tasks in the OpenML suite identified by suite_id. For any tasks that are not yet in the repo,
    inspect_openml_task is invoked on the task. Any tasks that succeed and can be added to the repo are placed in
    succeeded_tasks, and any that fail (and possibly require manual inspection or importing) are added on failed_tasks.

    Args:
        suite_id: integer describing the suite ID

    Returns:
        succeeded_tasks, failed_tasks: list of tasks that pass the tests, and list of tasks that do not pass the tests
            (as task IDs). Only tasks that are not already in the repo are included.
    """
    implemented_tasks = {kwargs["openml_task_id"] for kwargs in openml_tasks}

    suite = openml.study.get_suite(suite_id)
    task_list = suite.tasks

    failed_tasks = []
    succeeded_tasks = []

    for openml_task_id in task_list:
        if openml_task_id in implemented_tasks:
            continue
        errors = inspect_openml_task(openml_task_id, debug=False)
        if errors:
            failed_tasks.append(openml_task_id)
        else:
            succeeded_tasks.append(openml_task_id)

    return succeeded_tasks, failed_tasks


# Call the dataset preprocessor decorator for each of the selected OpenML datasets
for kwargs in openml_tasks:
    # if kwargs["openml_task_id"] in [48, 50]:
    #    continue
    task = openml.tasks.get_task(
        task_id=kwargs["openml_task_id"], download_data=False, download_qualities=False
    )
    ds = openml.datasets.get_dataset(
        task.dataset_id, download_data=False, download_qualities=False
    )
    dataset_name = f"openml__{ds.name}__{kwargs['openml_task_id']}"

    dataset_preprocessor(preprocessor_dict, dataset_name, generate_split=False)(
        functools.partial(preprocess_openml, **kwargs)
    )
