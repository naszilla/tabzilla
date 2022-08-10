import openml
import numpy as np

from tabzilla_datasets import TabularDataset

cv_n_folds = 10

openml_tasks = [
    {
        "openml_task_id": 361089,
        "dataset_name": "California_OpenML",
        "cat_mask_patch": None,
        "target_type": "regression"
    },
    {
        "openml_task_id": 2071,
        "dataset_name": "Adult_OpenML",
        "cat_mask_patch": None,
        "target_type": "binary"
    },
]

# Based on: https://github.com/releaunifreiburg/WellTunedSimpleNets/blob/main/utilities.py
# TODO: Determine form of cat_mask_patch and implement functionality
# TODO: Implement val
def preprocess_openml(dataset_name, openml_task_id, target_type, cat_mask_patch=None):
    # task_id = 233088
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

    dataset = task.get_dataset()
    X, y, categorical_indicator, _ = dataset.get_data(
        dataset_format='dataframe',
        target=dataset.default_target_attribute,
    )
    if target_type == "regression":
        num_classes = 1
    elif target_type == "binary":
        num_classes = 1
    elif target_type == "classification":
        num_classes = len(y.unique())
    else:
        raise RuntimeError(f"Unrecognized target_type: {target_type}")

    cat_idx = [idx for idx, indicator in enumerate(categorical_indicator) if indicator]

    dataset = TabularDataset(dataset_name, X.values, np.array(y),
                             cat_idx=cat_idx,
                             target_type=target_type,
                             num_classes=num_classes,
                             split_indeces=split_indeces)
    return dataset
