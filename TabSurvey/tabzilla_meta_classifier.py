from ast import Str
import numpy as np
# Model imports
from sklearn.metrics import accuracy_score, precision_score
from sklearn.multioutput import RegressorChain, MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.svm import SVR

from tabzilla_meta_featurizer import get_cached_featurized
from tabzilla_meta_metrics import get_metrics

METADATA_VERSION = "v0"
METADATA_PATH = f"./temp/metafeatures_{METADATA_VERSION}.csv"

def get_metalearner(name: str):
    """
    Args:
        name (str): type of metalearner

    Returns:
        model class that supports model.fit and model.predict
    """
    if name == 'linear':
        model = Pipeline([("scaler", StandardScaler()),
                         ("linear", MultiOutputRegressor(Ridge(alpha=10))),
                          ])
    elif name == 'knn':
        n_neighbors = 5
        model = Pipeline([("scaler", StandardScaler()),
                         ("knn", KNeighborsRegressor(n_neighbors=n_neighbors))])
    elif name == 'random':
        raise NotImplementedError("Need to implement this class")
    
    return model


if __name__ == '__main__':
    # Load the meta data
    X_train, y_train, X_test, y_test, y_best_test = get_cached_featurized(METADATA_PATH)

    # fetch the model 
    model = get_metalearner('linear')

    # Fit the model
    model.fit(X_train, y_train)

    # Get predictions and metrics
    preds = model.predict(X_test)
    metrics = get_metrics(y_test, y_best_test, preds)

    print("Metrics: ")
    print(metrics)
