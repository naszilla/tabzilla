import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

# TODO: Handle validation?
def process_data(dataset, train_index, test_index, verbose=False, scale=False, one_hot_encode=False):
    num_mask = np.ones(dataset.X.shape[1])
    num_mask[dataset.cat_idx] = 0
    # TODO: Remove this assertion after sufficient testing
    assert num_mask.sum() + len(dataset.cat_idx) == dataset.X.shape[1]


    X_train, y_train = dataset.X[train_index], dataset.y[train_index]
    X_test, y_test = dataset.X[test_index], dataset.y[test_index]

    if scale:
        if verbose:
            print("Scaling the data...")
        scaler = StandardScaler()
        X_train[:, num_mask] = scaler.fit_transform(X_train[:, num_mask])
        X_test[:, num_mask] = scaler.transform(X_test[:, num_mask])

    if one_hot_encode:
        ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
        new_x1 = ohe.fit_transform(X_train[:, dataset.cat_idx])
        X_train = np.concatenate([new_x1, X_train[:, num_mask]], axis=1)
        new_x1_test = ohe.transform(X_test[:, dataset.cat_idx])
        X_test = np.concatenate([new_x1_test, X_test[:, num_mask]], axis=1)
        if verbose:
            print("New Shape:", X_train.shape)

    return {"data_train": (X_train, y_train),
            "data_test": (X_test, y_test)}