# Running Experiments

Each "experiment" refers to an (algorithm, dataset) pair: during the experiment we run at most 30 train/test cycles for an algorithm on a dataset, where the first cycle uses the algorithm's default hyperparameters, and the following 29 are randomly sampled hyperparameters.

Each experiment has a 10-hour limit, and each train/test cycle has a 2-hour limit (these are specified in the files `tabzilla_experiment_config.yml` and `tabzilla_experiment_config_gpu.yml`, for GPU and CPU experiments, respectively).

To run these experiments in reasonable batches, we partition the set of algorithms and datasets into sets, each specified in a bash file. The sets of algorithms include:
- `ALGS_CPU_1.sh`
- `ALGS_CPU_2.sh`
- `ALGS_GPU_1.sh`
- `ALGS_GPU_2.sh`

And the sets of datasets are:
- `DATASETS_A.sh`
- `DATASETS_B.sh`

## Batch Scripts

We use a separate batch script for each pair of (algorithm-set, dataset-set). For example, the script `experiemnts/algs_cpu_1_datasets_b.sh` iterates over all algorithms from `ALGS_CPU_1.sh` with all datasets from `DATASETS_B.sh`.

Each batch script is associated with a different "experiment name", which is how we keep track of all experiments that have been run. The list of all batch scripts that have been run, and their experiment names, are listed in the file `experiment_log.txt`.

