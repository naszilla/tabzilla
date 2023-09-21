import logging
import argparse
import itertools
import multiprocessing
from pathlib import Path

import pandas as pd

from tabzilla_results_aggregator import parse_results_file

# Parse results
def parse_results(dest_folder):
    results_files = dest_folder.glob("*_results.json")
    result_list = []
    exception_list = []
    for results_file in results_files:
        new_results, new_exceptions = parse_results_file(results_file, "")
        result_list += new_results
        exception_list += new_exceptions
    return result_list, exception_list

def consolidate_results(args):
    logging.info("Begin parsing results directories...")
    base_directory = Path(args.base_directory)
    child_directories = list(base_directory.glob("*"))
    with multiprocessing.Pool(processes=args.num_processes) as pool:
        results_and_exceptions = pool.map(parse_results, child_directories)

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

        if args.out_results_file.exists():
            old_results = pd.read_csv(args.out_results_file)
            consolidated_results = pd.concat(
                [old_results, consolidated_results], axis=0, ignore_index=True
            )

        consolidated_results.sort_values(
            ["dataset_fold_id", "alg_hparam_id"], inplace=True
        )
        consolidated_results.to_csv(args.out_results_file, index=False)

    if exceptions:
        exceptions = pd.DataFrame(exceptions)

        if args.out_errors_file.exists():
            old_results = pd.read_csv(args.out_errors_file)
            exceptions = pd.concat([old_results, exceptions], axis=0, ignore_index=True)

        exceptions.sort_values(["dataset_name", "alg_hparam_id"], inplace=True)
        exceptions.to_csv(args.out_errors_file, index=False)
        logging.info("Done")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_directory", 
        type=str,
        default="/scratch/bf996/tabzilla/TabZilla/res",
        required=True,
        help="Base directory for results"
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=1,
        help="Number of processes to use for parsing results",
    )
    parser.add_argument(
        "--out_results_file",
        type=str,
        default="consolidated_results.csv",
        help="Path to output file for results",
    )
    parser.add_argument(
        "--out_errors_file",
        type=str,
        default="consolidated_errors.csv",
        help="Path to output file for errors",
    )
    args = parser.parse_args()
    args.out_results_file = Path(args.out_results_file)
    args.out_errors_file = Path(args.out_errors_file)
    consolidate_results(args)