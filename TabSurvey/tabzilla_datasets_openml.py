import openml
import numpy as np

from tabzilla_datasets import TabularDataset

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
def preprocess_openml(dataset_name, openml_task_id, target_type, cat_mask_patch=None):
    # test_size = 0.2, seed = 42
    # task_id = 233088
    task = openml.tasks.get_task(task_id=openml_task_id)
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

    # X_train, X_test, y_train, y_test = train_test_split(
    #     X,
    #     y,
    #     test_size=test_size,
    #     random_state=seed,
    #     #stratify=y,
    #     shuffle=True,
    # )

    dataset = TabularDataset(dataset_name, X.values, np.array(y),
                             cat_idx=cat_idx,
                             target_type=target_type,
                             num_classes=num_classes)
    return dataset
