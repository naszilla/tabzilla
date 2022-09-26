import pandas as pd
import itertools
import matplotlib.pyplot as plt


is_max_metric = {
    "Log Loss": False,
    "AUC": True,
    "Accuracy": True,
    "F1": True,
    "MSE": False,
    "R2": False
}

def get_metadata(suffix="_v0"):
    metadataset_df = pd.read_csv(f"../TabSurvey/metadataset{suffix}.csv")
    metafeatures_df = pd.read_csv(f"../TabSurvey/metafeatures{suffix}.csv")
    
    return metadataset_df, metafeatures_df

def process_metafeatures(metafeatures_df, filter_families=None):
    """Impute missing values and filter by groups of features
    filter_families: list of strings from
    ['landmarking',
    'general',
    'statistical',
    'model-based',
    'info-theory',
    'relative',
    'clustering', # Not currently implemente; OOM
    # 'complexity', # Not currently implemente; OOM
    # 'itemset', # Not currently implemente; OOM
    # 'concept' # Not currently implemented
    ]
    """
    metafeatures_processed = metafeatures_df.fillna(metafeatures_df.median())
    
    if filter_families is not None:
        prefixes = [f"f__pymfe.{family}" for family in filter_families]
        filter_cols = [col for col in metafeatures_processed.columns 
                       if not col.startswith("f__") or any(col.startswith(prefix) for prefix in prefixes)]
        metafeatures_processed = metafeatures_processed[filter_cols]
    
    
    return metafeatures_processed
    
    

def get_tuned_alg_perf(metadataset_df, metric="Accuracy"):
    """ For each algorithm, for each dataset fold, tune on "validation". You should analyze the performance on "test" after this. """
    if metric not in is_max_metric:
        raise RuntimeError(f"metric must be one of: {is_max_metric.keys()}")
    groups = metadataset_df.groupby(["alg_name", "dataset_fold_id"])[f"{metric}__val"]
    
    if is_max_metric[metric]:
        idxopt = groups.idxmax()
    else:
        idxopt = groups.idxmin()
    
    tuned_alg_perf = metadataset_df.loc[idxopt.dropna()]
    return tuned_alg_perf


def compute_feature_corrs(joined_df, metric_name, as_abs=False):
    """
    Compute correlation between each metafeature and each algorithm (when tuned)
    """
    query_column = f"{metric_name}__test"
    all_cors = []

    all_features = [col for col in joined_df.columns if col.startswith("f__")]

    for alg, filtered_results in joined_df.groupby("alg_name"):
        alg_cors = filtered_results[all_features].corrwith(filtered_results[f"{metric_name}__test"])
        alg_cors.name = alg
        all_cors.append(alg_cors)
    all_cors = pd.concat(all_cors, axis=1)
    if as_abs:
        all_cors = all_cors.abs()
    return all_cors

