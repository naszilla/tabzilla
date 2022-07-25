from pathlib import Path
import argparse
import sklearn
import pandas as pd
import functools

from tabzilla_datasets import TabularDataset

dataset_path = Path("datasets")

preprocessors = {}

def dataset_preprocessor(dataset_name, target_encode=False, cat_feature_encode=True):
    """
    Adds the function to the dictionary of pre-processors, which can then be called as preprocessors[dataset_name]()
    Args:
        dataset_name: Name of the dataset

    """
    def dataset_preprocessor_decorator(func):
        @functools.wraps(func)
        def wrapper_preprocessor(*args, **kwargs):
            dataset = func(*args, **kwargs)
            if target_encode:
                dataset.target_encode()
            if cat_feature_encode:
                dataset.cat_feature_encode()
            return dataset

        if dataset_name in preprocessors:
            raise RuntimeError(f"Duplicate dataset names not allowed: {dataset_name}")
        preprocessors[dataset_name] = wrapper_preprocessor
        return wrapper_preprocessor

    return dataset_preprocessor_decorator


def preprocess_dataset(dataset_name, overwrite=False):
    print(f"Processing {dataset_name}...")
    dest_path = dataset_path / dataset_name
    if not overwrite and dest_path.exists():
        print(f"Found existing folder {dest_path}. Skipping.")
        return
    dataset = preprocessors[dataset_name](dataset_name)
    dataset.write(dest_path, overwrite=overwrite)
    return


@dataset_preprocessor("CaliforniaHousing", target_encode=False)
def preprocess_cal_housing(dataset_name):
    X, y = sklearn.datasets.fetch_california_housing(return_X_y=True)
    dataset = TabularDataset(dataset_name, X, y,
                             cat_idx=[],
                             target_type="regression",
                             num_classes=1)
    return dataset

@dataset_preprocessor("Covertype", target_encode=True)
def preprocess_covertype(dataset_name):
    X, y = sklearn.datasets.fetch_covtype(return_X_y=True)
    dataset = TabularDataset(dataset_name, X, y,
                             cat_idx=[],
                             target_type="classification",
                             num_classes=7)
    return dataset

@dataset_preprocessor("Adult", target_encode=True)
def preprocess_covertype(dataset_name):
    url_data = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

    features = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital-status', 'occupation',
                'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
    label = "income"
    columns = features + [label]
    df = pd.read_csv(url_data, names=columns)

    df.fillna(0, inplace=True)

    X = df[features].to_numpy()
    y = df[label].to_numpy()

    dataset = TabularDataset(dataset_name, X, y,
                             cat_idx=[1,3,5,6,7,8,9,13],
                             target_type="binary",
                             num_classes=1)
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset pre-processing utility.")
    parser.add_argument('--dataset_name',
                        help="Dataset to pre-process.")
    parser.add_argument('--process_all', action='store_true',
                        help="Use this flag to pre-process all datasets.")
    parser.add_argument('--overwrite', action='store_true',
                        help="Use this flag to overwrite datasets.")

    args = parser.parse_args()

    if args.dataset_name is not None and args.process_all:
        print("dataset_name cannot be specified simultaneously with the flag process_all")

    else:
        if args.process_all:
            for dataset_name in preprocessors.keys():
                preprocess_dataset(dataset_name, args.overwrite)
        else:
            preprocess_dataset(args.dataset_name, args.overwrite)