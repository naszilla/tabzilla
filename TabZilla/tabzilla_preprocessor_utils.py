import functools

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from tabzilla_datasets import TabularDataset

cv_n_folds = 10  # Number of folds to use for splitting


def dataset_preprocessor(
    preprocessor_dict,
    dataset_name,
    target_encode=None,
    cat_feature_encode=True,
    generate_split=True,
):
    """
    Adds the function to the dictionary of pre-processors, which can then be called as preprocessor_dict[dataset_name]()
    Args:
        dataset_name: Name of the dataset

    """

    def dataset_preprocessor_decorator(func):
        @functools.wraps(func)
        def wrapper_preprocessor(*args, **kwargs):
            dataset_kwargs = func(*args, **kwargs)
            if generate_split:
                dataset_kwargs["split_indeces"] = split_dataset(dataset_kwargs)
                dataset_kwargs["split_source"] = "random_init"
            dataset = TabularDataset(dataset_name, **dataset_kwargs)

            # Infer target_encode based on target type
            is_regression = dataset.target_type == "regression"
            if (target_encode is None and not is_regression) or target_encode:
                dataset.target_encode()
            if cat_feature_encode:
                dataset.cat_feature_encode()
            return dataset

        if dataset_name in preprocessor_dict:
            raise RuntimeError(f"Duplicate dataset names not allowed: {dataset_name}")
        preprocessor_dict[dataset_name] = wrapper_preprocessor
        return wrapper_preprocessor

    return dataset_preprocessor_decorator


def split_dataset(dataset_kwargs, num_splits=cv_n_folds, shuffle=True, seed=0):
    target_type = dataset_kwargs["target_type"]

    if target_type == "regression":
        kf = KFold(n_splits=num_splits, shuffle=shuffle, random_state=seed)
    elif target_type == "classification" or target_type == "binary":
        kf = StratifiedKFold(n_splits=num_splits, shuffle=shuffle, random_state=seed)
    else:
        raise NotImplementedError("Objective" + target_type + "is not yet implemented.")

    splits = kf.split(dataset_kwargs["X"], dataset_kwargs["y"])

    split_indeces = []
    for train_indices, test_indices in splits:
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

    return split_indeces
