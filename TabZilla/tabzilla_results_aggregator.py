# NOTE: you need to authenticate the google cloud CLI before using this script: https://cloud.google.com/docs/authentication/provide-credentials-adc#how-to

import argparse
import itertools
import json
import logging
import multiprocessing
import warnings
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
try:
    from google.cloud import storage
except ImportError:
    print("Google cloud not installed")

logging.basicConfig(format="[%(asctime)s] : %(message)s", level=logging.INFO)

local_results_folder = Path("bucket_results")
save_results_folder = Path("results_files")
PROJECT_NAME = "research-collab-naszilla"
RESULTS_BUCKET_NAME = "tabzilla-results"
ROOT_PATH = "results"
out_results_file = Path("metadataset.csv")
out_errors_file = Path("metadataset_errors.csv")

num_processes = 2


def process_blob(args):
    i = args[0] + 1
    result_blob = args[1]
    num_blobs = args[2]
    logging.info(f"[blob {i} of {num_blobs}]: Processing...")
    blob_path = Path(result_blob)

    # if blob has already been downloaded, skip it
    dest_folder = save_results_folder / blob_path.relative_to(ROOT_PATH)

    # check whether dest folder exists and is populated
    if dest_folder.is_dir():
        logging.info(f"[blob {i} of {num_blobs}]: Already parsed")

        results_files = [
            x
            for x in dest_folder.iterdir()
            if x.is_file() and x.name.endswith("results.json")
        ]
        logging.info(
            f"[blob {i} of {num_blobs}]: found {len(results_files)} results files"
        )

        if len(results_files) == 0:
            raise Exception(f"results dir contains no results files: {dest_folder}")
    else:
        local_path = local_results_folder / blob_path.relative_to(ROOT_PATH)

        # Download and extract contents
        logging.info(f"[blob {i} of {num_blobs}]: Downloading...")
        local_path.parent.mkdir(parents=True, exist_ok=True)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            storage.Client(project=PROJECT_NAME).get_bucket(
                RESULTS_BUCKET_NAME
            ).get_blob(result_blob).download_to_filename(local_path)

        logging.info(f"[blob {i} {num_blobs}]: Extracting...")
        dest_folder.mkdir(parents=True, exist_ok=True)
        with ZipFile(local_path, "r") as zf:
            # extract only the "*_results.json" files
            extract_files = []
            for f in zf.namelist():
                if f.endswith("results.json"):
                    extract_files.append(f)

            logging.info(
                f"[blob {i} of {num_blobs}]: found {len(extract_files)} results files"
            )
            logging.info(f"[blob {i} of  {num_blobs}]: extracting...")
            for f in extract_files:
                zf.extract(f, dest_folder)

        # remove downloaded zip file
        local_path.unlink()

    # Parse results
    logging.info(f"[blob {i} of {num_blobs}]: Parsing...")
    results_files = dest_folder.glob("*_results.json")
    result_list = []
    exception_list = []
    for results_file in results_files:
        new_results, new_exceptions = parse_results_file_blob(results_file, blob_path)
        result_list += new_results
        exception_list += new_exceptions

    logging.info(f"[blob {i} of {num_blobs}]: Done!")

    return result_list, exception_list


def read_results_folder(results_folder: Path):
    """
    read all results files from a directory, and return a list of dicts, one for each result in the directory

    args:
    - results_folder: the directory with results files matching pattern *_results.json
    """
    results_files = results_folder.glob("*_results.json")
    result_list = []
    exception_list = []
    for results_file in results_files:
        new_results, new_exceptions = parse_results_file(results_file)
        result_list += new_results
        exception_list += new_exceptions

    return result_list, exception_list


def parse_results_to_csv(results_folder: Path):
    """read all results files from a folder using read_results_folder and write them to a csv file in the same directory named results.csv"""

    result_list, exception_list = read_results_folder(results_folder)

    results_file_path = None
    exceptions_file_path = None

    if result_list:
        consolidated_results = pd.DataFrame(result_list)

        consolidated_results.sort_values(
            ["dataset_fold_id", "alg_hparam_id"], inplace=True
        )
        consolidated_results.to_csv(results_file_path, index=False)
        results_file_path = results_folder / "results.csv"

    if exception_list:
        consolidated_exceptions = pd.DataFrame(exception_list)

        consolidated_exceptions.sort_values(
            ["dataset_name", "alg_hparam_id"], inplace=True
        )
        consolidated_exceptions.to_csv(exceptions_file_path, index=False)
        exceptions_file_path = results_folder / "exceptions.csv"

    return results_file_path, exceptions_file_path


def parse_results_file(results_file):
    result_list, exception_list = [], []

    with open(results_file, "r") as f:
        contents = json.load(f)

    dataset_name = contents["dataset"]["name"]
    alg_name = contents["model"]["name"]

    is_exception = contents["exception"] != "None"
    if not is_exception:
        num_folds = len(contents["timers"]["train"])
        for fold_number in range(num_folds):
            fold_results = dict(
                dataset_fold_id=f"{dataset_name}__fold_{fold_number}",
                dataset_name=dataset_name,
                target_type=contents["dataset"]["target_type"],
                alg_name=alg_name,
                hparam_source=contents["hparam_source"],
                trial_number=contents["trial_number"],
                alg_hparam_id=f'{alg_name}__seed_{contents["experiemnt_args"]["hparam_seed"]}__trial_{contents["trial_number"]}',
            )

            for phase in ["train", "val", "test", "train-eval"]:
                fold_results[f"time__{phase}"] = contents["timers"][phase][fold_number]
                if phase == "train-eval":
                    continue
                for metric in contents["scorers"][phase].keys():
                    fold_results[f"{metric}__{phase}"] = contents["scorers"][phase][
                        metric
                    ][fold_number]

            result_list.append(fold_results)
    else:
        exception_info = dict(
            dataset_name=dataset_name,
            alg_name=alg_name,
            hparam_source=contents["hparam_source"],
            trial_number=contents["trial_number"],
            alg_hparam_id=f'{alg_name}__seed_{contents["experiemnt_args"]["hparam_seed"]}__trial_{contents["trial_number"]}',
            exception=contents["exception"],
        )
        exception_list.append(exception_info)

    return result_list, exception_list


# this version is for processing results files downloaded as gcloud blobs
def parse_results_file_blob(results_file, blob_path):
    result_list, exception_list = [], []
    dataset_name = results_file.parent.parent.parent.name
    alg_name = results_file.parent.parent.name
    exp_name = results_file.parent.name

    with open(results_file, "r") as f:
        contents = json.load(f)

    is_exception = contents["exception"] != "None"
    try:
        blob_path_posix = blob_path.as_posix()
    except:
        blob_path_posix = blob_path
    if not is_exception:
        num_folds = len(contents["timers"]["train"])
        for fold_number in range(num_folds):
            fold_results = dict(
                results_bucket_path=blob_path_posix,
                dataset_fold_id=f"{dataset_name}__fold_{fold_number}",
                dataset_name=dataset_name,
                target_type=contents["dataset"]["target_type"],
                alg_name=contents["model"]["name"],
                hparam_source=contents["hparam_source"],
                trial_number=contents["trial_number"],
                alg_hparam_id=f'{alg_name}__seed_{contents["experiemnt_args"]["hparam_seed"]}__trial_{contents["trial_number"]}',
                exp_name=exp_name,
            )

            for phase in ["train", "val", "test", "train-eval"]:
                fold_results[f"time__{phase}"] = contents["timers"][phase][fold_number]
                if phase == "train-eval":
                    continue
                for metric in contents["scorers"][phase].keys():
                    fold_results[f"{metric}__{phase}"] = contents["scorers"][phase][
                        metric
                    ][fold_number]

            result_list.append(fold_results)
    else:
        exception_info = dict(
            results_bucket_path=blob_path_posix,
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


def download_and_process_results(args):
    processed_file_set = set()
    if out_results_file.exists():
        done_results = pd.read_csv(out_results_file)
        processed_file_set = set(done_results["results_bucket_path"])
        del done_results
    if out_errors_file.exists():
        done_errors = pd.read_csv(out_errors_file)
        processed_file_set = processed_file_set.union(
            set(done_errors["results_bucket_path"])
        )
        del done_errors

    local_results_folder.mkdir(parents=True, exist_ok=True)

    storage_client = storage.Client(project=PROJECT_NAME)
    matching_blobs = storage_client.list_blobs(RESULTS_BUCKET_NAME, prefix=ROOT_PATH)

    # don't process blobs that have already been processed
    matching_blobs = [
        blob.name for blob in matching_blobs if blob.name not in processed_file_set
    ]

    # only process blobs that meet this
    if args.blob_name_contains != "":
        matching_blobs = [x for x in matching_blobs if args.blob_name_contains in x]

    num_blobs = len(matching_blobs)
    args = [(i, blobname, num_blobs) for i, blobname in enumerate(matching_blobs)]
    with multiprocessing.Pool(processes=num_processes) as pool:
        results_and_exceptions = pool.map(process_blob, args)

    # Flatten list of lists
    exceptions = list(itertools.chain(*(exc for _, exc in results_and_exceptions)))
    consolidated_results = list(
        itertools.chain(*(res for res, _ in results_and_exceptions))
    )
    del results_and_exceptions

    if not consolidated_results and not exceptions:
        logging.info("No new results. Exiting.")
        return

    logging.info("Parsing done. Aggregating results...")
    if consolidated_results:
        consolidated_results = pd.DataFrame(consolidated_results)

        if out_results_file.exists():
            old_results = pd.read_csv(out_results_file)
            consolidated_results = pd.concat(
                [old_results, consolidated_results], axis=0, ignore_index=True
            )

        consolidated_results.sort_values(
            ["dataset_fold_id", "alg_hparam_id"], inplace=True
        )
        consolidated_results.to_csv(out_results_file, index=False)

    if exceptions:
        exceptions = pd.DataFrame(exceptions)

        if out_errors_file.exists():
            old_results = pd.read_csv(out_errors_file)
            exceptions = pd.concat([old_results, exceptions], axis=0, ignore_index=True)

        exceptions.sort_values(["dataset_name", "alg_hparam_id"], inplace=True)
        exceptions.to_csv(out_errors_file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--blob_name_contains",
        type=str,
        default="",
        required=False,
        help="only download blobs with a name that contains this string",
    )

    args = parser.parse_args()

    download_and_process_results(args)
