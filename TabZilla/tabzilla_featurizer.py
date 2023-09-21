from pathlib import Path

import pandas as pd
from pymfe.mfe import MFE
from tabzilla_data_processing import process_data
from tabzilla_datasets import TabularDataset
from tqdm import tqdm

# Subset of: ['landmarking', 'general', 'statistical', 'model-based', 'info-theory', 'relative', 'clustering',
# 'complexity', 'itemset', 'concept']
groups = [
    "landmarking",
    "general",
    "statistical",
    "model-based",
    "info-theory",
    "relative",
    # 'clustering', # OOM
    # 'complexity', # OOM
    # 'itemset', # OOM
    # 'concept'
]


# Subset of:
# ["mean", "sd", "count", "histogram", "iq_range", "kurtosis", "max", "median", "min", "quantiles", "range",
# "skewness"]
summary_funcs = [
    "mean",
    "sd",
    "count",
    "histogram",
    "iq_range",
    "kurtosis",
    "max",
    "median",
    "min",
    "quantiles",
    "range",
    "skewness",
]

# One of: ["accuracy", "balanced-accuracy", "f1", "kappa", "auc"]
# For landmarking
scoring = "balanced-accuracy"


def featurize_dataset(dataset_path):
    dataset = TabularDataset.read(dataset_path)
    if dataset.target_type not in ["classification", "binary"]:
        print("Unsupported target type. Skipping.")
        return None

    print("Processing...")
    metafeats = []
    for fold_idx, split in tqdm(enumerate(dataset.split_indeces)):
        # Process data (handles imputation)
        processed_data = process_data(
            dataset,
            split["train"],
            [0],
            [0],
            scaler="None",
            one_hot_encode=False,
            impute=True,
            args=None,
        )
        X_train, y_train = processed_data["data_train"]

        # Extract metafeatures
        mfe = MFE(groups=groups, summary=summary_funcs, random_state=0, score=scoring)

        mfe.fit(
            X_train,
            y_train,
            cat_cols=dataset.cat_idx,
            transform_num=False,
            transform_cat=None,
        )
        ft = mfe.extract()

        # Consolidate results
        fold_metafeats = {"dataset_name": f"{dataset.name}__fold_{fold_idx}"}
        for group in groups:
            ft_group = mfe.parse_by_group(group, ft)
            fold_metafeats.update(
                {f"f__pymfe.{group}.{name}": value for name, value in zip(*ft_group)}
            )
            # metafeats += [{"group": group, "name": name, "value": value} for name, value in zip(*ft_group)]
        metafeats.append(fold_metafeats)
    return metafeats


def featurize_all_datasets():
    data_path = Path("datasets")

    output_file = Path("metafeatures.csv")
    if output_file.exists():
        computed_features = pd.read_csv(output_file)
        computed_features.set_index("dataset_name", inplace=True)
    else:
        computed_features = None

    for dataset_path in data_path.glob("*"):
        sample_name = f"{dataset_path.name}__fold_0"
        if computed_features is not None and sample_name in computed_features.index:
            continue

        print(dataset_path)
        # dataset = TabularDataset.read(Path("datasets/openml__covertype__7593"))

        dataset_metafeatures = featurize_dataset(dataset_path)
        if dataset_metafeatures is None:
            continue

        dataset_metafeatures = pd.DataFrame(dataset_metafeatures)
        dataset_metafeatures.set_index("dataset_name", inplace=True)

        if computed_features is None:
            computed_features = dataset_metafeatures
            computed_features = computed_features[sorted(computed_features.columns)]
        else:
            computed_features = pd.concat([dataset_metafeatures, computed_features])
        print("Writing. Do not interrupt...")
        computed_features.to_csv(output_file)


featurize_all_datasets()
