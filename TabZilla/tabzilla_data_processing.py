import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer


def process_data(
    dataset,
    train_index,
    val_index,
    test_index,
    verbose=False,
    scaler="None",
    one_hot_encode=False,
    impute=True,
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

    return {
        "data_train": (X_train, y_train),
        "data_val": (X_val, y_val),
        "data_test": (X_test, y_test),
    }
