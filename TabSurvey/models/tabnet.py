import numpy as np
import torch
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
from models.tabnet_patch_lib.tabnet_patch import TabNetClassifierPatched
from utils.io_utils import load_model_from_file, save_model_to_file

from models.basemodel_torch import BaseModelTorch

"""
    TabNet: Attentive Interpretable Tabular Learning (https://arxiv.org/pdf/1908.07442.pdf)

    See the implementation: https://github.com/dreamquark-ai/tabnet
"""


class TabNet(BaseModelTorch):
    def __init__(self, params, args):
        super().__init__(params, args)

        # Paper recommends to be n_d and n_a the same
        self.params["n_a"] = self.params["n_d"]

        self.params["cat_idxs"] = args.cat_idx
        self.params["cat_dims"] = args.cat_dims

        # TabZilla: remove device from params, since it is never used
        # self.params["device_name"] = self.device

        if args.objective == "regression":
            self.model = TabNetRegressor(**self.params)
            self.metric = ["rmse"]
        elif args.objective == "classification" or args.objective == "binary":
            #self.model = TabNetClassifier(**self.params)
            self.model = TabNetClassifierPatched(**self.params)
            self.metric = ["logloss"]
        
    def fit(self, X, y, X_val=None, y_val=None):
        if self.args.objective == "regression":
            y, y_val = y.reshape(-1, 1), y_val.reshape(-1, 1)
        elif self.args.objective == "binary":
            self.model.num_classes = 2
        elif self.args.objective == "classification":
            self.model.num_classes = self.args.num_classes

        # Drop last only if last batch has only one sample
        drop_last = X.shape[0] % self.args.batch_size == 1
        self.model.fit(
            X,
            y,
            eval_set=[(X_val, y_val)],
            eval_name=["eval"],
            eval_metric=self.metric,
            max_epochs=self.args.epochs,
            patience=self.args.early_stopping_rounds,
            batch_size=self.args.batch_size,
            drop_last=drop_last,
        )
        history = self.model.history
        self.save_model(filename_extension="best")
        return history["loss"], history["eval_" + self.metric[0]]

    def predict_helper(self, X):
        X = np.array(X, dtype=float)

        if self.args.objective == "regression":
            return self.model.predict(X)
        elif self.args.objective == "classification" or self.args.objective == "binary":
            return self.model.predict_proba(X)

    def save_model(self, filename_extension=""):
        save_model_to_file(self.model, self.args, filename_extension)

    def load_model(self, filename_extension=""):
        self.model = load_model_from_file(self.model, self.args, filename_extension)

    def get_model_size(self):
        # To get the size, the model has be trained for at least one epoch
        model_size = sum(
            t.numel() for t in self.model.network.parameters() if t.requires_grad
        )
        return model_size

    @classmethod
    def define_trial_parameters(cls, trial, args):
        params = {
            "n_d": trial.suggest_int("n_d", 8, 64),
            "n_steps": trial.suggest_int("n_steps", 3, 10),
            "gamma": trial.suggest_float("gamma", 1.0, 2.0),
            "cat_emb_dim": trial.suggest_int("cat_emb_dim", 1, 3),
            "n_independent": trial.suggest_int("n_independent", 1, 5),
            "n_shared": trial.suggest_int("n_shared", 1, 5),
            "momentum": trial.suggest_float("momentum", 0.001, 0.4, log=True),
            "mask_type": trial.suggest_categorical(
                "mask_type", ["sparsemax", "entmax"]
            ),
        }
        return params

    # TabZilla: add function for seeded random params and default params
    @classmethod
    def get_random_parameters(cls, seed):
        rs = np.random.RandomState(seed)
        params = {
            "n_d": rs.randint(8, 65),
            "n_steps": rs.randint(3, 11),
            "gamma": 1.0 + rs.rand(),
            "cat_emb_dim": rs.randint(1, 4),
            "n_independent": rs.randint(1, 6),
            "n_shared": rs.randint(1, 6),
            "momentum": 0.4 * np.power(10, rs.uniform(-3, -1)),
            "mask_type": rs.choice(["sparsemax", "entmax"]),
        }
        return params

    @classmethod
    def default_parameters(cls):
        params = {
            "n_d": 24,
            "n_steps": 5,
            "gamma": 1.5,
            "cat_emb_dim": 2,
            "n_independent": 3,
            "n_shared": 3,
            "momentum": 0.015,
            "mask_type": "sparsemax",
        }
        return params

    def attribute(self, X: np.ndarray, y: np.ndarray, stategy=""):
        """Generate feature attributions for the model input.
        Only strategy are supported: default ("")
        Return attribution in the same shape as X.
        """
        X = np.array(X, dtype=float)
        attributions = self.model.explain(torch.tensor(X, dtype=torch.float32))[0]
        return attributions
