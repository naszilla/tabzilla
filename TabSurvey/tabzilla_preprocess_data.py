from pathlib import Path
import argparse
import sklearn

from tabzilla_datasets import TabularDataset

dataset_path = Path("datasets")

preprocessors = {}

def registered_preprocessor(dataset_name):
    """
    Adds the function to the dictionary of pre-processors, which can then be called as preprocessors[dataset_name]()
    Args:
        dataset_name: Name of the dataset

    """
    def register_func_decorator(func):
        preprocessors[dataset_name] = lambda: func(dataset_name)
        return func
    return register_func_decorator


def preprocess_dataset(dataset_name, overwrite=False):
    print(f"Processing {dataset_name}...")
    dest_path = dataset_path / dataset_name
    if not overwrite and dest_path.exists():
        print(f"Found existing folder {dest_path}. Skipping.")
        return
    dataset = preprocessors[dataset_name]()
    dataset.write(dest_path)
    return


@registered_preprocessor("CaliforniaHousing")
def preprocess_cal_housing(dataset_name):
    X, y = sklearn.datasets.fetch_california_housing(return_X_y=True)
    dataset = TabularDataset(dataset_name, X, y, [], "regression", 1, 8, len(y))
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