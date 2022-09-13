import argparse
from pathlib import Path
import json
from zipfile import ZipFile
from tqdm import tqdm

import pandas as pd
from google.cloud import storage


local_results_folder = Path("bucket_results")
PROJECT_NAME = 'research-collab-naszilla'
RESULTS_BUCKET_NAME = "tabzilla-results"
ROOT_PATH = "inbox"

def download_results():
    # Download all files
    print("Downloading files...")
    local_results_folder.mkdir(parents=True, exist_ok=True)

    storage_client = storage.Client(project=PROJECT_NAME)
    for result_blob in storage_client.list_blobs(RESULTS_BUCKET_NAME, prefix=ROOT_PATH):
        blob_path = Path(result_blob.name)
        local_path = local_results_folder / blob_path.relative_to(ROOT_PATH)
        if not local_path.exists():
            print(blob_path)
            print("Downloading...")
            local_path.parent.mkdir(parents=True, exist_ok=True)
            result_blob.download_to_filename(local_path)

            print("Extracting...")
            with ZipFile(local_path, 'r') as zf:
                zf.extractall(local_path.with_suffix(""))

    print("Finished!")


def aggregate_results():
    consolidated_results = []

    for results_file in tqdm(local_results_folder.glob("*/*/*/*.json")):
        dataset_name = results_file.parent.parent.parent.name
        alg_name = results_file.parent.parent.name
        exp_name = results_file.parent.name
        result_set = results_file.name

        with open(results_file, "r") as f:
            contents = json.load(f)

        for fold_number in range(len(contents["splits"])):
            fold_results = dict(
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

            consolidated_results.append(fold_results)

    consolidated_results = pd.DataFrame(consolidated_results)

    consolidated_results.to_csv("metadataset.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregator for TabZilla results.")
    parser.add_argument('--action',
                        choices=['download', 'aggregate'],
                        help='Action to perform.')
    args = parser.parse_args()

    if args.action == "download":
        download_results()
    elif args.action == "aggregate":
        aggregate_results()
