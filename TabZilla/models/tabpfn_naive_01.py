from tabpfn import TabPFNClassifier
from models.basemodel import BaseModel
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import numpy as np
import math

MAX_TABPFN_FEATURES = 100
MAX_TABPFN_SAMPLES = 1000

class TabPFNModel(BaseModel):
    def __init__(self, params, args):
        super().__init__(params, args)

        if args.objective == "regression":
            raise NotImplementedError("Does not support")
        elif args.objective == "classification":
            self.model = TabPFNClassifier(device='cuda', N_ensemble_configurations=32)
        elif args.objective == "binary":
            self.model = TabPFNClassifier(device='cuda', N_ensemble_configurations=32)

    def fit(self, X, y, X_val=None, y_val=None):
        if X.shape[0] > MAX_TABPFN_SAMPLES:
            indices = np.random.choice(X.shape[0], MAX_TABPFN_SAMPLES, replace=False)
            X = X[indices]
            y = y[indices]
        if X.shape[1] > MAX_TABPFN_FEATURES:
            self.selector = SelectKBest(mutual_info_classif, k=MAX_TABPFN_FEATURES)
            X_new = self.selector.fit_transform(X, y)
            X = X_new
        else:
            pass
        return super().fit(X, y)
        
    def predict_proba(self, X):
        if X.shape[1] > MAX_TABPFN_FEATURES:
            X_new = self.selector.transform(X)
            X = X_new
        else:
            pass
        if X.shape[0] > MAX_TABPFN_SAMPLES:
            X_ens = []
            X_preds = []
            for idx, i in enumerate(range(0, X.shape[0], MAX_TABPFN_SAMPLES)):
                print(f"Fitting samples {idx+1} of {math.ceil(X.shape[0]/MAX_TABPFN_SAMPLES)}")
                X_ens.append(X[i:i+MAX_TABPFN_SAMPLES])
                X_preds.append(self.model.predict_proba(X_ens[-1]))
            self.prediction_probabilities = np.concatenate(X_preds, axis=0)
        else:
            self.prediction_probabilities = self.predict_helper(X)
        # If binary task returns only probability for the true class, adapt it to return (N x 2)
        if self.prediction_probabilities.shape[1] == 1:
            self.prediction_probabilities = np.concatenate(
                (1 - self.prediction_probabilities, self.prediction_probabilities), 1
            )
        return self.prediction_probabilities

    def predict_helper(self, X):
        return self.model.predict_proba(X)

    @classmethod
    def define_trial_parameters(cls, trial, args):
        params = dict()
        return params

    @classmethod
    def get_random_parameters(cls, seed: int):
        params = dict()
        return params

    @classmethod
    def default_parameters(cls):
        params = dict()
        return params

    def get_classes(self):
        return self.model.classes_

