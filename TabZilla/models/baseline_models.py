import random

import numpy as np
from sklearn import ensemble, linear_model, neighbors, svm, tree

from models.basemodel import BaseModel

"""
    Define all Models implemented by the Sklearn library: 
    Linear Model, KNN, SVM, Decision Tree, Random Forest
"""

"""
    Linear Model - Ordinary least squares Linear Regression / Logistic Regression
    
    Takes no hyperparameters
"""

# TabZilla: add function to generate seeded random parameters, and default parameters.


class LinearModel(BaseModel):
    def __init__(self, params, args):
        super().__init__(params, args)

        if args.objective == "regression":
            self.model = linear_model.LinearRegression(n_jobs=-1)
        elif args.objective == "classification":
            self.model = linear_model.LogisticRegression(
                multi_class="multinomial", n_jobs=-1
            )
        elif args.objective == "binary":
            self.model = linear_model.LogisticRegression(n_jobs=-1)

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


"""
    K-Neighbors Regressor - Regression/Classification based on k-nearest neighbors
    
    Takes number of neighbors as hyperparameters
"""


class KNN(BaseModel):
    def __init__(self, params, args):
        super().__init__(params, args)

        if args.objective == "regression":
            self.model = neighbors.KNeighborsRegressor(
                n_neighbors=params["n_neighbors"],
                algorithm=params["knn_alg"],
                leaf_size=params["leaf_size"],
                n_jobs=-1,
            )
        elif args.objective == "classification" or args.objective == "binary":
            self.model = neighbors.KNeighborsClassifier(
                n_neighbors=params["n_neighbors"],
                algorithm=params["knn_alg"],
                leaf_size=params["leaf_size"],
                n_jobs=-1,
            )

    def fit(self, X, y, X_val=None, y_val=None):
        return super().fit(X, y, X_val, y_val)

    @classmethod
    def define_trial_parameters(cls, trial, args):
        params = {
            "n_neighbors": trial.suggest_categorical(
                "n_neighbors", list(range(3, 42, 2))
            ),
            "knn_alg": trial.suggest_categorical("knn_alg", ["kd_tree", "ball_tree"]),
            "leaf_size": trial.suggest_int("leaf_size", [30, 50, 70, 100, 300]),
        }
        return params

    @classmethod
    def get_random_parameters(cls, seed: int):
        rs = np.random.RandomState(seed)
        params = {
            "n_neighbors": 1 + 2 * rs.randint(1, 21),
            "knn_alg": rs.choice(["kd_tree", "ball_tree"]),
            "leaf_size": rs.choice([30, 50, 70, 100, 300]),
        }
        return params

    @classmethod
    def default_parameters(cls):
        params = {
            "n_neighbors": 9,
            "knn_alg": "kd_tree",
            "leaf_size": 30,
        }
        return params

    def get_classes(self):
        return self.model.classes_

"""
    Support Vector Machines - Epsilon-Support Vector Regression / C-Support Vector Classification
    
    Takes the regularization parameter as hyperparameter
"""


class SVM(BaseModel):
    def __init__(self, params, args):
        super().__init__(params, args)

        if args.objective == "regression":
            self.model = svm.SVR(C=params["C"])
        elif args.objective == "classification" or args.objective == "binary":
            self.model = svm.SVC(C=params["C"], probability=True)

    @classmethod
    def define_trial_parameters(cls, trial, args):
        params = {"C": trial.suggest_float("C", 1e-10, 1e10, log=True)}
        return params

    @classmethod
    def get_random_parameters(cls, seed: int):
        rs = np.random.RandomState(seed)
        params = {"C": np.power(10, rs.uniform(-10, 10))}
        return params

    @classmethod
    def default_parameters(cls):
        params = {"C": 1.0}
        return params

    def get_classes(self):
        return self.model.classes_


"""
    Decision Tree - Decision Tree Regressor/Classifier
    
    Takes the maximum depth of the tree as hyperparameter
"""


class DecisionTree(BaseModel):
    def __init__(self, params, args):
        super().__init__(params, args)

        if args.objective == "regression":
            self.model = tree.DecisionTreeRegressor(max_depth=params["max_depth"])
        elif args.objective == "classification" or args.objective == "binary":
            self.model = tree.DecisionTreeClassifier(max_depth=params["max_depth"])

    @classmethod
    def define_trial_parameters(cls, trial, args):
        params = {"max_depth": trial.suggest_int("max_depth", 2, 12, log=True)}
        return params

    @classmethod
    def get_random_parameters(cls, seed: int):
        rs = np.random.RandomState(seed)
        params = {"max_depth": int(np.round(np.power(2, rs.uniform(1, np.log2(12)))))}
        return params

    @classmethod
    def default_parameters(cls):
        params = {"max_depth": 5}
        return params

    def get_classes(self):
        return self.model.classes_

"""
    Random Forest - Random Forest Regressor/Classifier
    
    Takes the maximum depth of the trees and the number of estimators as hyperparameter
"""


class RandomForest(BaseModel):
    def __init__(self, params, args):
        super().__init__(params, args)

        if args.objective == "regression":
            self.model = ensemble.RandomForestRegressor(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                n_jobs=-1,
            )
        elif args.objective == "classification" or args.objective == "binary":
            self.model = ensemble.RandomForestClassifier(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                n_jobs=-1,
            )

    @classmethod
    def define_trial_parameters(cls, trial, args):
        params = {
            "max_depth": trial.suggest_int("max_depth", 2, 12, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 5, 100, log=True),
        }
        return params

    @classmethod
    def get_random_parameters(cls, seed: int):
        rs = np.random.RandomState(seed)
        params = {
            "max_depth": int(np.round(np.power(2, rs.uniform(1, np.log2(12))))),
            "n_estimators": int(
                np.round(np.power(5, rs.uniform(1, np.log2(100) / np.log2(5))))
            ),
        }
        return params

    @classmethod
    def default_parameters(cls):
        params = {
            "max_depth": 5,
            "n_estimators": 50,
        }
        return params

    def get_classes(self):
        return self.model.classes_