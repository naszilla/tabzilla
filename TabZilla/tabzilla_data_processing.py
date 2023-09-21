import time

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer


class SubsetMaker(object):
    def __init__(
        self, subset_features, subset_rows, subset_features_method, subset_rows_method
    ):
        self.subset_features = subset_features
        self.subset_rows = subset_rows
        self.subset_features_method = subset_features_method
        self.subset_rows_method = subset_rows_method
        self.row_selector = None
        self.feature_selector = None

    def random_subset(self, X, y, action=[]):
        if "rows" in action:
            row_indices = np.random.choice(X.shape[0], self.subset_rows, replace=False)
        else:
            row_indices = np.arange(X.shape[0])
        if "features" in action:
            feature_indices = np.random.choice(
                X.shape[1], self.subset_features, replace=False
            )
        else:
            feature_indices = np.arange(X.shape[1])
        return X[row_indices[:, None], feature_indices], y[row_indices]

    def first_subset(self, X, y, action=[]):
        if "rows" in action:
            row_indices = np.arange(self.subset_rows)
        else:
            row_indices = np.arange(X.shape[0])
        if "features" in action:
            feature_indices = np.arange(self.subset_features)
        else:
            feature_indices = np.arange(X.shape[1])
        return X[row_indices[:, None], feature_indices], y[row_indices]

    def mutual_information_subset(self, X, y, action="features", split="train"):
        if split not in ["train", "val", "test"]:
            raise ValueError("split must be 'train', 'val', or 'test'")
        if split == "train":
            # NOTE: we are only fitting on the first split we see to save time here
            if getattr(self, "feature_selector", None) is None:
                print("Fitting mutual information feature selector ...")
                # start the timer
                timer = time.time()
                self.feature_selector = SelectKBest(
                    mutual_info_classif, k=self.subset_features
                )
                X = self.feature_selector.fit_transform(X, y)
                print(
                    f"Done fitting mutual information feature selector in {round(time.time() - timer, 1)} seconds"
                )
            else:
                X = self.feature_selector.transform(X)
            return X, y
        else:
            X = self.feature_selector.transform(X)
            return X, y

    def make_subset(
        self,
        X,
        y,
        split="train",
        seed=0,
    ):
        """
        Make a subset of the data matrix X, with subset_features features and subset_rows rows.
        :param X: data matrix
        :param y: labels
        :param subset_features: number of features to keep
        :param subset_rows: number of rows to keep
        :param subset_features_method: method to use for selecting features
        :param subset_rows_method: method to use for selecting rows
        :return: subset of X, y
        """
        np.random.seed(seed)

        if X.shape[1] > self.subset_features > 0:
            print(
                f"making {self.subset_features}-sized subset of {X.shape[1]} features ..."
            )
            if self.subset_features_method == "random":
                X, y = self.random_subset(X, y, action=["features"])
            elif self.subset_features_method == "first":
                X, y = self.first_subset(X, y, action=["features"])
            elif self.subset_features_method == "mutual_information":
                X, y = self.mutual_information_subset(
                    X, y, action="features", split=split
                )
            else:
                raise ValueError(
                    f"subset_features_method not recognized: {self.subset_features_method}"
                )
        if X.shape[0] > self.subset_rows > 0:
            print(f"making {self.subset_rows}-sized subset of {X.shape[0]} rows ...")
            if self.subset_rows_method == "random":
                X, y = self.random_subset(X, y, action=["rows"])
            elif self.subset_rows_method == "first":
                X, y = self.first_subset(X, y, action=["rows"])
            else:
                raise ValueError(
                    f"subset_rows_method not recognized: {self.subset_rows_method}"
                )

        return X, y


def process_data(
    dataset,
    train_index,
    val_index,
    test_index,
    verbose=False,
    scaler="None",
    one_hot_encode=False,
    impute=True,
    args=None,
):
    # validate the scaler
    assert scaler in ["None", "Quantile"], f"scaler not recognized: {scaler}"

    if scaler == "Quantile":
        scaler_function = QuantileTransformer(
            n_quantiles=min(len(train_index), 1000)
        )  # use either 1000 quantiles or num. training instances, whichever is smaller

    num_mask = np.ones(dataset.X.shape[1], dtype=int)
    num_mask[dataset.cat_idx] = 0
    # TODO: Remove this assertion after sufficient testing
    assert num_mask.sum() + len(dataset.cat_idx) == dataset.X.shape[1]

    X_train, y_train = dataset.X[train_index], dataset.y[train_index]
    X_val, y_val = dataset.X[val_index], dataset.y[val_index]
    X_test, y_test = dataset.X[test_index], dataset.y[test_index]

    # Impute numerical features
    if impute:
        num_idx = np.where(num_mask)[0]

        # The imputer drops columns that are fully NaN. So, we first identify columns that are fully NaN and set them to
        # zero. This will effectively drop the columns without changing the column indexing and ordering that many of
        # the functions in this repository rely upon.
        fully_nan_num_idcs = np.nonzero(
            (~np.isnan(X_train[:, num_idx].astype("float"))).sum(axis=0) == 0
        )[0]
        if fully_nan_num_idcs.size > 0:
            X_train[:, num_idx[fully_nan_num_idcs]] = 0
            X_val[:, num_idx[fully_nan_num_idcs]] = 0
            X_test[:, num_idx[fully_nan_num_idcs]] = 0

        # Impute numerical features, and pass through the rest
        numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer())])
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, num_idx),
                ("pass", "passthrough", dataset.cat_idx),
                # ("cat", categorical_transformer, categorical_features),
            ],
            # remainder="passthrough",
        )
        X_train = preprocessor.fit_transform(X_train)
        X_val = preprocessor.transform(X_val)
        X_test = preprocessor.transform(X_test)

        # Re-order columns (ColumnTransformer permutes them)
        perm_idx = []
        running_num_idx = 0
        running_cat_idx = 0
        for is_num in num_mask:
            if is_num > 0:
                perm_idx.append(running_num_idx)
                running_num_idx += 1
            else:
                perm_idx.append(running_cat_idx + len(num_idx))
                running_cat_idx += 1
        assert running_num_idx == len(num_idx)
        assert running_cat_idx == len(dataset.cat_idx)
        X_train = X_train[:, perm_idx]
        X_val = X_val[:, perm_idx]
        X_test = X_test[:, perm_idx]

    if scaler != "None":
        if verbose:
            print(f"Scaling the data using {scaler}...")
        X_train[:, num_mask] = scaler_function.fit_transform(X_train[:, num_mask])
        X_val[:, num_mask] = scaler_function.transform(X_val[:, num_mask])
        X_test[:, num_mask] = scaler_function.transform(X_test[:, num_mask])

    if one_hot_encode:
        ohe = OneHotEncoder(sparse=False, handle_unknown="ignore")
        new_x1 = ohe.fit_transform(X_train[:, dataset.cat_idx])
        X_train = np.concatenate([new_x1, X_train[:, num_mask]], axis=1)
        new_x1_test = ohe.transform(X_test[:, dataset.cat_idx])
        X_test = np.concatenate([new_x1_test, X_test[:, num_mask]], axis=1)
        new_x1_val = ohe.transform(X_val[:, dataset.cat_idx])
        X_val = np.concatenate([new_x1_val, X_val[:, num_mask]], axis=1)
        if verbose:
            print("New Shape:", X_train.shape)

    # create subset of dataset if needed
    if (
        args is not None
        and (args.subset_features > 0 or args.subset_rows > 0)
        and (
            args.subset_features < args.num_features or args.subset_rows < len(X_train)
        )
    ):
        print(
            f"making subset with {args.subset_features} features and {args.subset_rows} rows..."
        )
        if getattr(dataset, "ssm", None) is None:
            dataset.ssm = SubsetMaker(
                args.subset_features,
                args.subset_rows,
                args.subset_features_method,
                args.subset_rows_method,
            )
        X_train, y_train = dataset.ssm.make_subset(
            X_train,
            y_train,
            split="train",
            seed=dataset.subset_random_seed,
        )
        if args.subset_features < args.num_features:
            X_val, y_val = dataset.ssm.make_subset(
                X_val,
                y_val,
                split="val",
                seed=dataset.subset_random_seed,
            )
            X_test, y_test = dataset.ssm.make_subset(
                X_test,
                y_test,
                split="test",
                seed=dataset.subset_random_seed,
            )
        print("subset created")

    return {
        "data_train": (X_train, y_train),
        "data_val": (X_val, y_val),
        "data_test": (X_test, y_test),
    }
