from pathlib import Path

import pandas as pd

# type of algorithms (neural, baseline, or gbdt)
ALG_TYPES = {
    "MLP": "neural",
    "TabNet": "neural",
    "VIME": "neural",
    "TabTransformer": "neural",
    "NODE": "neural",
    "STG": "neural",
    "NAM": "neural",
    "DeepFM": "neural",
    "SAINT": "neural",
    "DANet": "neural",
    "MLP-rtdl": "neural",
    "ResNet": "neural",
    "FTTransformer": "neural",
    "TabPFN": "pfn",
    "LightGBM": "gbdt",
    "XGBoost": "gbdt",
    "CatBoost": "gbdt",
    "DecisionTree": "baseline",
    "LinearModel": "baseline",
    "KNN": "baseline",
    "SVM": "baseline",
    "RandomForest": "baseline",
}

# display names for each alg
ALG_DISPLAY_NAMES = {
    "MLP": "MLP",
    "TabNet": "TabNet",
    "VIME": "VIME",
    "TabTransformer": "TabTransformer",
    "NODE": "NODE",
    "STG": "STG",
    "NAM": "NAM",
    "DeepFM": "DeepFM",
    "SAINT": "SAINT",
    "DANet": "DANet",
    "rtdl_MLP": "MLP-rtdl",
    "rtdl_ResNet": "ResNet",
    "rtdl_FTTransformer": "FTTransformer",
    "TabPFNModel": "TabPFN",
    "LightGBM": "LightGBM",
    "XGBoost": "XGBoost",
    "CatBoost": "CatBoost",
    "DecisionTree": "DecisionTree",
    "LinearModel": "LinearModel",
    "KNN": "KNN",
    "SVM": "SVM",
    "RandomForest": "RandomForest",
}


# Maps metric name to a boolean indicating whether maximizing the metric equates to having a better model
is_max_metric = {
    "Log Loss": False,
    "AUC": True,
    "Accuracy": True,
    "F1": True,
    "MSE": False,
    "R2": True,
}

metadata_folder = Path("../TabSurvey")


def get_metadata(suffix="_v0"):
    """
    Read the metadataset and metafeatures for the specified suffix.
    metadataset_df: as output by tabzilla_results_aggregator.py
    metafeatures_df: as output by tabzilla_featurizer.py

    Args:
        suffix: The suffix indicating the dataset (usually the version)

    Returns:
        metadataset_df, metafeatures_df: Tuple of DataFrames
    """
    metadataset_df = pd.read_csv(metadata_folder / f"metadataset{suffix}.csv")
    metafeatures_df = pd.read_csv(metadata_folder / f"metafeatures{suffix}.csv")

    return metadataset_df, metafeatures_df


def process_metafeatures(metafeatures_df, filter_families=None):
    """
    Impute missing values (with median of metafeatures) and filter by groups of features.

    Args:
        metafeatures_df: DataFrame, as output by get_metadata
        filter_families: list of strings from:
            ['landmarking',
            'general',
            'statistical',
            'model-based',
            'info-theory',
            'relative',
            'clustering', # Not currently implemented; OOM
            # 'complexity', # Not currently implemented; OOM
            # 'itemset', # Not currently implemented; OOM
            # 'concept' # Not currently implemented
            ]

    Returns:
        metafeatures_processed: dataframe similar to metafeatures_df, with imputed features and filtered by metafeature
            categories
    """
    metafeatures_processed = metafeatures_df.fillna(
        metafeatures_df.median(numeric_only=True)
    )

    if filter_families is not None:
        prefixes = [f"f__pymfe.{family}" for family in filter_families]
        filter_cols = [
            col
            for col in metafeatures_processed.columns
            if not col.startswith("f__")
            or any(col.startswith(prefix) for prefix in prefixes)
        ]
        metafeatures_processed = metafeatures_processed[filter_cols]

    return metafeatures_processed


def get_tuned_alg_perf(metadataset_df, metric="Accuracy", group_col="alg_name"):
    """For each algorithm, for each dataset fold, tune on "validation". Only one row per algorithm / dataset_fold_id
    pair will be kept. You should analyze the performance on "test" after this, with the same metric used to tune.

    Args:
        metadataset_df: DataFrame, as output by get_metadata
        metric: Metric name (choices as in the keys of is_max_metric)
        group_on: column to group rows for each algorithm. all rows with the same value of group_col will be treated as the same algorithm.

    Returns:
        tuned_alg_perf: dataframe with tuned performance
    """

    """  """
    if metric not in is_max_metric:
        raise RuntimeError(f"metric must be one of: {is_max_metric.keys()}")
    groups = metadataset_df.groupby([group_col, "dataset_fold_id"])[f"{metric}__val"]

    if is_max_metric[metric]:
        idxopt = groups.idxmax()
    else:
        idxopt = groups.idxmin()

    tuned_alg_perf = metadataset_df.loc[idxopt.dropna()]
    return tuned_alg_perf


def join_tuned_with_metafeatures(tuned_alg_perf, metafeatures_df):
    """
    Join the tuned performance dataframe with the metafeatures dataframe
    Args:
        tuned_alg_perf: DataFrame of tuned performance, as output by get_tuned_alg_perf()
        metafeatures_df: DataFrame of metafeatures

    Returns:

    """
    joined_df = tuned_alg_perf.merge(
        metafeatures_df, right_on="dataset_name", left_on="dataset_fold_id", how="left"
    )
    return joined_df


def compute_feature_corrs(joined_df, metric_name, as_abs=False):
    """

    Args:
        joined_df: As output by join_tuned_with_metafeatures()
        metric_name: The name of the metric. Use one of the keys of is_max_metric.
        as_abs: Boolean specifying whether to get the absolute values of all correlations.

    Returns:
        all_cors: DataFrame with one row per metafeature and one column per algorithm
    """

    query_column = f"{metric_name}__test"
    all_cors = []

    all_features = [col for col in joined_df.columns if col.startswith("f__")]

    for alg, filtered_results in joined_df.groupby("alg_name"):
        alg_cors = filtered_results[all_features].corrwith(
            filtered_results[query_column]
        )
        alg_cors.name = alg
        all_cors.append(alg_cors)
    all_cors = pd.concat(all_cors, axis=1)
    if as_abs:
        all_cors = all_cors.abs()
    return all_cors
