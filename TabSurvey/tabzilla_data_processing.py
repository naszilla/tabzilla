import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

# TODO: Handle validation?
def process_data(dataset, train_index, test_index, verbose=False, scale=False, one_hot_encode=False):
    num_idx = []
    cat_dims = []

    X = dataset.X
    y = dataset.y

    # TODO: Check this is done at the right place
    # Preprocess data
    for i in range(dataset.num_features):
        if dataset.cat_idx and i in dataset.cat_idx:
            le = LabelEncoder()
            X[:, i] = le.fit_transform(X[:, i])

            # Setting this?
            cat_dims.append(len(le.classes_))

        else:
            num_idx.append(i)

    X_train, y_train = X[train_index], y[train_index]
    X_test, y_test = X[test_index], y[test_index]

    if scale:
        if verbose:
            print("Scaling the data...")
        scaler = StandardScaler()
        X_train[:, num_idx] = scaler.fit_transform(X_train[:, num_idx])
        X_test[:, num_idx] = scaler.transform(X_test[:, num_idx])

    if one_hot_encode:
        ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
        new_x1 = ohe.fit_transform(X_train[:, dataset.cat_idx])
        X_train = np.concatenate([new_x1, X_train[:, num_idx]], axis=1)
        new_x1_test = ohe.transform(X_test[:, dataset.cat_idx])
        X_test = np.concatenate([new_x1_test, X_test[:, num_idx]], axis=1)
        if verbose:
            print("New Shape:", X.shape)

    return {"data_train": (X_train, y_train),
            "data_test": (X_test, y_test),
            "cat_dims": cat_dims}