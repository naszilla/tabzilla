import openml
import numpy as np
import functools

from tabzilla_preprocessor_utils import dataset_preprocessor, cv_n_folds

preprocessor_dict = {}

easy_import_task_file = "openml_easy_import_list.txt" # Datasets identified just by their ID can be easily imported from here

debug_mode = False

openml_tasks = [
    # This dataset has been added to openml_easy_import_list.txt as an example
    # {
    #     "openml_task_id": 361089,
    #     # "dataset_name": "openml_california", # Can be explicitly specified for faster execution
    #     # "target_type": "regression", # Does not need to be explicitly specified, but can be
    # },
    {
        "openml_task_id": 2071,
        # "dataset_name": "openml_adult", # Can be explicitly specified for faster execution
        # "target_type": "binary", # Does not need to be explicitly specified, but can be
        # "force_cat_features": ["workclass", "education"], # Example (these are not needed in this case)
        # "force_num_features": ["fnlwgt", "education-num"], # Example (these are not needed in this case)
    },
]

with open(easy_import_task_file, "r") as f:
    for line in f:
        task_id = int(line.strip())
        openml_tasks.append({"openml_task_id": task_id})


# Based on: https://github.com/releaunifreiburg/WellTunedSimpleNets/blob/main/utilities.py
def preprocess_openml(openml_task_id, target_type=None, force_cat_features=None, force_num_features=None):
    if force_num_features is None:
        force_num_features = []
    if force_cat_features is None:
        force_cat_features = []

    task = openml.tasks.get_task(task_id=openml_task_id)
    n_repeats, n_folds, n_samples = task.get_split_dimensions()
    if n_repeats != 1 or n_folds != cv_n_folds or n_samples != 1:
        raise NotImplementedError(f"Re-splitting required for split dimensions {n_repeats}, {n_folds}, {n_samples}.")

    # Extract splits
    split_indeces = []
    for split_idx in range(cv_n_folds):
        train_indices, test_indices = task.get_train_test_split_indices(repeat=0, fold=split_idx)
        split_indeces.append({
            "train": train_indices,
            "test": test_indices,
            "val": []
        })
    # Build validation set by using n+1 th test set.
    for split_idx in range(cv_n_folds):
        split_indeces[split_idx]["val"] = split_indeces[(split_idx+1) % cv_n_folds]["test"].copy()
        split_indeces[split_idx]["train"] = np.setdiff1d(split_indeces[split_idx]["train"],
                                                         split_indeces[split_idx]["val"], assume_unique=True)

    if debug_mode:
    # Sanity check
        for split_idx in range(cv_n_folds):
            print("Dimensions: {}, {}, {}".format(len(split_indeces[split_idx]["train"]),
                                                  len(split_indeces[split_idx]["val"]),
                                                  len(split_indeces[split_idx]["test"])))
            assert len(np.intersect1d(split_indeces[split_idx]["train"], split_indeces[split_idx]["test"])) == 0
            assert len(np.intersect1d(split_indeces[split_idx]["train"], split_indeces[split_idx]["val"])) == 0
            assert len(np.intersect1d(split_indeces[split_idx]["test"], split_indeces[split_idx]["val"])) == 0
            for split_jdx in range(split_idx+1, cv_n_folds):
                intersect = np.intersect1d(split_indeces[split_idx]["test"], split_indeces[split_jdx]["test"])
                assert len(intersect) == 0

    dataset = task.get_dataset()
    X, y, categorical_indicator, col_names = dataset.get_data(
        dataset_format='dataframe',
        target=task.target_name,
    )

    # Infer task type if not provided
    if target_type is None:
        if task.task_type == "Supervised Regression":
            target_type = "regression"
        elif task.task_type == "Supervised Classification":
            n_unique_labels = len(task.class_labels)
            if n_unique_labels == 2:
                target_type = "binary"
            elif n_unique_labels > 2:
                target_type = "classification"
            else:
                raise RuntimeError(f"Unexpected number of class labels: {n_unique_labels}")
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

    # Patch categorical columns and numerical columns in case of any mislabeling
    for feature_name in force_cat_features:
        categorical_indicator[col_names.index(feature_name)] = True
    for feature_name in force_num_features:
        categorical_indicator[col_names.index(feature_name)] = False

    cat_idx = [idx for idx, indicator in enumerate(categorical_indicator) if indicator]

    return {
        "X": X.values,
        "y": np.array(y),
        "cat_idx": cat_idx,
        "target_type": target_type,
        "num_classes": num_classes,
        "split_indeces": split_indeces,
        "split_source": "openml",
    }


# Call the dataset preprocessor decorator for each of the selected OpenML datasets
for kwargs in openml_tasks:
    kwargs_copy = {key: val for key, val in kwargs.items() if key != "dataset_name"}
    # Create a dataset name if not specified
    if not "dataset_name" in kwargs:
        task = openml.tasks.get_task(task_id=kwargs["openml_task_id"], download_data=False, download_qualities=False)
        ds = openml.datasets.get_dataset(task.dataset_id, download_data=False, download_qualities=False)
        dataset_name = f"openml_{ds.name}"
    else:
        dataset_name = kwargs["dataset_name"]
    dataset_preprocessor(preprocessor_dict, dataset_name, generate_split=False)(
        functools.partial(preprocess_openml, **kwargs_copy))