# Analyzing TabZilla Results

## Listing all results

We can start by listing all of the experiment results, which are stored on a GCP bucket. This is handled by the script [`TabSurvey/tabzilla_result_logger.py`](TabSurvey/tabzilla_result_logger.py), which produces the file [`TabSurvey/result_log.txt`](TabSurvey/result_log.txt):

```
python tabzilla_result_logger.py 
```

The file `result_log.txt` is a CSV that contains one row for each experiment result file in our GCP bucket (there are currently ~3920 experiment result files listed here). 

For example, the first few rows are:

```
dataset,alg,experiment
openml__APSFailure__168868,CatBoost,gpu-expt-a
openml__APSFailure__168868,DANet,algs-gpu-2-datasets-a
openml__APSFailure__168868,DecisionTree,cpu-expt
...
```

## Aggregating Performance Results

We use the script [`TabSurvey/tabzilla_results_aggregator.py`](TabSurvey/tabzilla_results_aggregator.py) to aggregate all performance results. This script loops over all compressed results files in the GCP bucket (the same results files listed in `result_log.txt`, as in the previous section), extracts them, and aggregates the performance results into two files: `metadataset.csv` (contains performance metrics) and `metadataset_errors.csv` (contains any exceptions caught during the experiments).

This script can filter only result files that contain a certain string, with the flag `--blob_name_contains`. For example:

```
python tabzilla_results_aggregator.py --blob_name_contains algs-gpu-1-datasets-b
```

will extract all result files with filenames including "algs-gpu-1-datasets-b".

**Note:** this script will check for results already included in `metadataset.csv`, and will skip these results files.

##  Initial Analysis

Our first analysis is a sanity-check of each **batch** of experiments (see the README in folder `scripts`). For each batch we have two files: `review_results_*.ipynb`, and `review_errors_*.ipynb`, where the "*" is replaced by the experiment batch name.

