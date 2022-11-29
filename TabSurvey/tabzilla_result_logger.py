# create a file experiment_index.txt of all result zip files that have been created

import itertools
import json
import logging
import multiprocessing
import shutil
import warnings
from pathlib import Path
from zipfile import ZipFile
import argparse

import pandas as pd
from google.cloud import storage

logging.basicConfig(format="[%(asctime)s] : %(message)s", level=logging.INFO)

PROJECT_NAME = "research-collab-naszilla"
RESULTS_BUCKET_NAME = "tabzilla-results"
ROOT_PATH = "results"
LOGFILE_PATH = "result_log.txt"

def create_log():

    storage_client = storage.Client(project=PROJECT_NAME)
    all_blobs = storage_client.list_blobs(RESULTS_BUCKET_NAME, prefix=ROOT_PATH)

    dataset_list = []
    alg_list = []
    expt_list = []

    for blob in all_blobs:
        blob_name = blob.name.split('/')
        dataset_list.append(blob_name[1])
        alg_list.append(blob_name[2])

        # remove the timestamp and random string from the filename
        if len(blob_name[3]) < 23:
            expt_list.append(blob_name[3])
        else:
            expt_list.append(blob_name[3][:-23])

    with open(LOGFILE_PATH, 'w') as f:
        f.write("dataset,alg,experiment\n")
        for i in range(len(dataset_list)):
            f.write(f"\n{dataset_list[i]},{alg_list[i]},{expt_list[i]}")

if __name__ == "__main__":
    create_log()
