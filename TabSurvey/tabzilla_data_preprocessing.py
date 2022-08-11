import argparse
from pathlib import Path

# Import all preprocessor modules and add them to list for them to be in list of preprocessors
import tabzilla_preprocessors_openml
import tabzilla_preprocessors
preprocessor_modules = [tabzilla_preprocessors, tabzilla_preprocessors_openml]

dataset_path = Path("datasets")

def build_preprocessors_dict():
    preprocessors = {}
    for module in preprocessor_modules:
        # TODO: Safe handling for duplicate keys
        preprocessors.update(module.preprocessor_dict)
    return preprocessors

preprocessors = build_preprocessors_dict()


def preprocess_dataset(dataset_name, overwrite=False):
    print(f"Processing {dataset_name}...")
    dest_path = dataset_path / dataset_name
    if not overwrite and dest_path.exists():
        print(f"Found existing folder {dest_path}. Skipping.")
        return
    if dataset_name not in preprocessors:
        raise KeyError(f"Unrecognized dataset name: {dataset_name}")
    dataset = preprocessors[dataset_name]()
    dataset.write(dest_path, overwrite=overwrite)
    return


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
        raise RuntimeError("dataset_name cannot be specified simultaneously with the flag process_all")
    elif args.dataset_name is None and not args.process_all:
        raise RuntimeError("Need to specify either dataset_name or process_all flag")

    if args.process_all:
        for dataset_name in preprocessors.keys():
            preprocess_dataset(dataset_name, args.overwrite)
    else:
        preprocess_dataset(args.dataset_name, args.overwrite)