import argparse
from pathlib import Path
import json
from zipfile import ZipFile
import multiprocessing
import itertools
import shutil
import functools
import warnings

import pandas as pd
from google.cloud import storage


local_results_folder = Path("bucket_results")
PROJECT_NAME = 'research-collab-naszilla'
RESULTS_BUCKET_NAME = "tabzilla-results"
ROOT_PATH = "results"
out_results_file = Path("metadataset.csv")
out_errors_file = Path("metadataset_errors.csv")

num_processes = 8


def process_blob(result_blob):
    blob_path = Path(result_blob)

    # Download and extract contents
    local_path = local_results_folder / blob_path.relative_to(ROOT_PATH)
    print(f"Downloading: {blob_path}...")
    local_path.parent.mkdir(parents=True, exist_ok=True)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        storage.Client(project=PROJECT_NAME).get_bucket(RESULTS_BUCKET_NAME).get_blob(result_blob).download_to_filename(local_path)

    print(f"Extracting: {blob_path}...")
    dest_folder = local_path.with_suffix("")
    with ZipFile(local_path, 'r') as zf:
        zf.extractall(dest_folder)
    local_path.unlink()

    # Parse results
    print(f"Parsing: {blob_path}...")
    results_files = dest_folder.glob("*_results.json")
    result_list = []
    exception_list = []
    for results_file in results_files:
        new_results, new_exceptions = parse_results_file(results_file, blob_path)
        result_list += new_results
        exception_list += new_exceptions

    # Clean up
    shutil.rmtree(dest_folder)
    print(f"Done!: {blob_path}...")

    return result_list, exception_list


def parse_results_file(results_file, blob_path):
    # print(results_file)

    result_list, exception_list = [], []
    dataset_name = results_file.parent.parent.parent.name
    alg_name = results_file.parent.parent.name
    exp_name = results_file.parent.name
    result_set = results_file.name

    with open(results_file, "r") as f:
        contents = json.load(f)

    is_exception = contents["exception"] != "None"
    if not is_exception:
        num_folds = len(contents['timers']['train'])
        for fold_number in range(num_folds):
            fold_results = dict(
                results_bucket_path=blob_path.as_posix(),
                dataset_fold_id=f"{dataset_name}__fold_{fold_number}",
                dataset_name=dataset_name,
                alg_name=contents["model"]["name"],
                hparam_source=contents["hparam_source"],
                trial_number=contents["trial_number"],
                alg_hparam_id=f'{alg_name}__seed_{contents["experiemnt_args"]["hparam_seed"]}__trial_{contents["trial_number"]}',
                exp_name=exp_name
            )

            for phase in ["train", "val", "test", "train-eval"]:
                fold_results[f"time__{phase}"] = contents["timers"][phase][fold_number]
                if phase == "train-eval":
                    continue
                for metric in contents["scorers"][phase].keys():
                    fold_results[f"{metric}__{phase}"] = contents["scorers"][phase][metric][fold_number]

            result_list.append(fold_results)
    else:
        exception_info = dict(
            results_bucket_path=blob_path.as_posix(),
            dataset_name=dataset_name,
            alg_name=contents["model"]["name"],
            hparam_source=contents["hparam_source"],
            trial_number=contents["trial_number"],
            alg_hparam_id=f'{alg_name}__seed_{contents["experiemnt_args"]["hparam_seed"]}__trial_{contents["trial_number"]}',
            exp_name=exp_name,
            exception=contents["exception"],
        )
        exception_list.append(exception_info)

    return result_list, exception_list


def download_and_process_results():
    processed_file_set = set()
    if out_results_file.exists():
        done_results = pd.read_csv(out_results_file)
        processed_file_set = set(done_results["results_bucket_path"])
        del done_results
    if out_errors_file.exists():
        done_errors = pd.read_csv(out_errors_file)
        processed_file_set = processed_file_set.union(set(done_errors["results_bucket_path"]))
        del done_errors

    local_results_folder.mkdir(parents=True, exist_ok=True)

    storage_client = storage.Client(project=PROJECT_NAME)
    matching_blobs = storage_client.list_blobs(RESULTS_BUCKET_NAME, prefix=ROOT_PATH)
    matching_blobs = [blob.name for blob in matching_blobs if blob.name not in processed_file_set]

    with multiprocessing.Pool(processes=num_processes) as pool:
        results_and_exceptions = pool.map(process_blob, matching_blobs)
    shutil.rmtree(local_results_folder)

    # Flatten list of lists
    exceptions = list(itertools.chain(*(exc for _, exc in results_and_exceptions)))
    consolidated_results = list(itertools.chain(*(res for res, _ in results_and_exceptions)))
    del results_and_exceptions

    if not consolidated_results and not exceptions:
        print("No new results. Exiting.")
        return

    print("Parsing done. Aggregating results...")
    if consolidated_results:
        consolidated_results = pd.DataFrame(consolidated_results)

        if out_results_file.exists():
            old_results = pd.read_csv(out_results_file)
            consolidated_results = pd.concat([old_results, consolidated_results], axis=0, ignore_index=True)

        consolidated_results.sort_values(["dataset_fold_id", "alg_hparam_id"], inplace=True)
        consolidated_results.to_csv(out_results_file, index=False)

    if exceptions:
        exceptions = pd.DataFrame(exceptions)

        if out_errors_file.exists():
            old_results = pd.read_csv(out_errors_file)
            exceptions = pd.concat([old_results, exceptions], axis=0, ignore_index=True)

        exceptions.sort_values(["dataset_name", "alg_hparam_id"], inplace=True)
        exceptions.to_csv(out_errors_file, index=False)

if __name__ == "__main__":
    download_and_process_results()