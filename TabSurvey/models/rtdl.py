# model classes for each of the three models used in
# Revisiting Deep Learning Models for Tabular Data (Gorishniy et al.)
# paper: https://arxiv.org/abs/2106.11959
# code: https://yura52.github.io (this is based on the notebook example)
#
# there are three models implemented for this paper
# rtdl_MLP (a custom MLP for tabular data, similar to the other MLP from the original TabSurvey repo)
# rtdl_ResNet (a custom version of ResNet for tabular data)
# FTTransformer (their new method)
#
# NOTE: this implementation: uses the training & inference functions from TabSurvey's BaseModelTorch class

from typing import Any, List

import numpy as np
import torch
import torch.nn as nn

import rtdl
from models.basemodel_torch import BaseModelTorch
import torch.nn.functional as F

class RTDL_MLP_Model(nn.Module):
    """model class for the rtdl MLP"""

    D_TOKEN = 8  # size of the embedding for a single feature
    LAYERS = [128, 256, 128]
    DROPOUT = 0.1

    def __init__(
        self,
        num_idx: List[Any],
        cat_idx: List[Any],
        cat_dims: List[Any],
        d_out: int,
        task: str,
    ):
        super().__init__()

        self.task = task

        self.has_cat_features = len(cat_idx) > 0

        if self.has_cat_features:
            self.cat_tokenizer = rtdl.CategoricalFeatureTokenizer(
                cat_dims, self.D_TOKEN, False, "uniform"
            )
            self.model = rtdl.MLP.make_baseline(
                d_in=len(num_idx)
                + self.cat_tokenizer.n_tokens * self.cat_tokenizer.d_token,
                d_layers=self.LAYERS,
                dropout=self.DROPOUT,
                d_out=d_out,
            )
        else:
            self.cat_tokenizer = None
            self.model = rtdl.MLP.make_baseline(
                d_in=len(num_idx),
                d_layers=self.LAYERS,
                dropout=self.DROPOUT,
                d_out=d_out,
            )

        
        self.num_idx = num_idx  # indices of numerical variables
        self.cat_idx = cat_idx  # indices of categorical variables
        self.cat_dims = cat_dims  # numer of levels in each categorical variable

    # define forward to apply tokenizer
    def forward(self, x):

        x_num = x[:, self.num_idx]

        if self.has_cat_features:
            x_cat = x[:, self.cat_idx].to(torch.int)
            x_ordered = torch.cat([x_num, self.cat_tokenizer(x_cat).flatten(1, -1)], dim=1)
        else:
            x_ordered = x_num

        if self.task == "classification":
            x = F.softmax(self.model(x_ordered), dim=1)
        else:
            x = self.model(x_ordered)

        return x


class RTDL_ResNet_Model(nn.Module):
    """model class for the rtdl ResNet"""

    D_TOKEN = 8  # size of the embedding for a single feature
    N_BLOCKS = 2
    D_MAIN = 128
    D_HIDDEN = 256
    DROPOUT_FIRST = 0.25
    DROPOUT_SECOND = 0.1

    def __init__(
        self,
        num_idx: List[Any],
        cat_idx: List[Any],
        cat_dims: List[Any],
        d_out: int,
        task: str,
    ):
        super().__init__()

        self.task = task

        self.has_cat_features = len(cat_idx) > 0

        if self.has_cat_features:
            self.cat_tokenizer = rtdl.CategoricalFeatureTokenizer(
                cat_dims, self.D_TOKEN, False, "uniform"
            )
            self.model = rtdl.ResNet.make_baseline(
                d_in=len(num_idx)
                + self.cat_tokenizer.n_tokens * self.cat_tokenizer.d_token,
                n_blocks=self.N_BLOCKS,
                d_main=self.D_MAIN,
                d_hidden=self.D_HIDDEN,
                dropout_first=self.DROPOUT_FIRST,
                dropout_second=self.DROPOUT_SECOND,
                d_out=d_out,
            )
        else:
            self.cat_tokenizer = None
            self.model = rtdl.ResNet.make_baseline(
                d_in=len(num_idx),
                n_blocks=self.N_BLOCKS,
                d_main=self.D_MAIN,
                d_hidden=self.D_HIDDEN,
                dropout_first=self.DROPOUT_FIRST,
                dropout_second=self.DROPOUT_SECOND,
                d_out=d_out,
            )

        self.num_idx = num_idx  # indices of numerical variables
        self.cat_idx = cat_idx  # indices of categorical variables
        self.cat_dims = cat_dims  # numer of levels in each categorical variable

    # define forward to apply tokenizer
    def forward(self, x):

        x_num = x[:, self.num_idx]

        if self.has_cat_features:
            x_cat = x[:, self.cat_idx].to(torch.int)
            x_ordered = torch.cat([x_num, self.cat_tokenizer(x_cat).flatten(1, -1)], dim=1)
        else:
            x_ordered = x_num

        if self.task == "classification":
            x = F.softmax(self.model(x_ordered), dim=1)
        else:
            x = self.model(x_ordered)
            
        return x


class RTDL_FTTransformer_Model(nn.Module):
    """model class for the rtdl FTTransformer. this returns the default model, with n_blocks=3"""

    def __init__(
        self,
        num_idx: List[Any],
        cat_idx: List[Any],
        cat_dims: List[Any],
        d_out: int,
        task: str,
    ):
        super().__init__()

        self.task = task

        self.model = rtdl.FTTransformer.make_default(
            n_num_features=len(num_idx),  # number of numerical features
            cat_cardinalities=cat_dims,  # number of levels of each categorical feature
            d_out=d_out,  # number of output heads
        )

        self.num_idx = num_idx  # indices of numerical variables
        self.cat_idx = cat_idx  # indices of categorical variables
        self.cat_dims = cat_dims  # numer of levels in each categorical variable

        self.has_cat_features = len(self.cat_idx) > 0
        self.has_num_features = len(self.num_idx) > 0

    # define forward, separating numerical and categorical features
    def forward(self, x):

        if self.has_num_features:
            x_num = x[:, self.num_idx]
        else:
            x_num = None

        if self.has_cat_features:
            x_cat = x[:, self.cat_idx].to(torch.int)
        else:
            x_cat = None        

        if self.task == "classification":
            x = F.softmax(self.model(x_num, x_cat), dim=1)
        else:
            x = self.model(x_num, x_cat)
            
        return x


class rtdl_MLP(BaseModelTorch):
    """default parameters. code is adapted from the rtdl example colab notebook"""

    objtype_not_implemented = ["regression"]

    def __init__(self, params, args):
        super().__init__(params, args)

        if args.objective == "regression":
            raise Exception("not implemented for regression yet.")

        num_mask = np.ones(args.num_features)
        num_mask[args.cat_idx] = 0
        num_idx = np.where(num_mask)[0]

        self.model = RTDL_MLP_Model(
            num_idx=num_idx,
            cat_idx=args.cat_idx,
            cat_dims=args.cat_dims,
            d_out=args.num_classes,
            task=args.objective,
        )

        self.to_device()

    # this is copied from TabSurvey's models.mlp.MLP
    def predict_helper(self, X):
        X = np.array(X, dtype=float)
        return super().predict_helper(X)

    @classmethod
    def define_trial_parameters(cls, trial, args):
        params = (
            {
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.0001, 0.01, log=True
                )
            },
        )
        return params

    @classmethod
    def get_random_parameters(cls, seed: int):
        rs = np.random.RandomState(seed)
        params = {"learning_rate": np.power(10, rs.uniform(-4, -2))}
        return params

    @classmethod
    def default_parameters(cls):
        params = {"learning_rate": 0.001}
        return params

    def fit(self, X, y, X_val=None, y_val=None):
        X = np.array(X, dtype=float)
        X_val = np.array(X_val, dtype=float)

        return super().fit(X, y, X_val, y_val)


class rtdl_ResNet(BaseModelTorch):
    """default parameters. code is adapted from the rtdl example colab notebook"""
    objtype_not_implemented = ["regression"]

    def __init__(self, params, args):
        super().__init__(params, args)

        if args.objective == "regression":
            raise Exception("not implemented for regression yet.")

        num_mask = np.ones(args.num_features)
        num_mask[args.cat_idx] = 0
        num_idx = np.where(num_mask)[0]

        self.model = RTDL_ResNet_Model(
            num_idx=num_idx,
            cat_idx=args.cat_idx,
            cat_dims=args.cat_dims,
            d_out=args.num_classes,
            task=args.objective,
        )

        self.to_device()

    # this is copied from TabSurvey's models.mlp.MLP
    def predict_helper(self, X):
        X = np.array(X, dtype=float)
        return super().predict_helper(X)

    @classmethod
    def define_trial_parameters(cls, trial, args):
        params = (
            {
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.0001, 0.01, log=True
                )
            },
        )
        return params

    @classmethod
    def get_random_parameters(cls, seed: int):
        rs = np.random.RandomState(seed)
        params = {"learning_rate": np.power(10, rs.uniform(-4, -2))}
        return params

    @classmethod
    def default_parameters(cls):
        params = {"learning_rate": 0.001}
        return params

    def fit(self, X, y, X_val=None, y_val=None):
        X = np.array(X, dtype=float)
        X_val = np.array(X_val, dtype=float)

        return super().fit(X, y, X_val, y_val)


class rtdl_FTTransformer(BaseModelTorch):
    """default parameters. code is adapted from the rtdl example colab notebook"""
    objtype_not_implemented = ["regression"]
    
    def __init__(self, params, args):
        super().__init__(params, args)

        if args.objective == "regression":
            raise Exception("not implemented for regression yet.")

        num_mask = np.ones(args.num_features)
        num_mask[args.cat_idx] = 0
        num_idx = np.where(num_mask)[0]

        self.model = RTDL_FTTransformer_Model(
            num_idx=num_idx,
            cat_idx=args.cat_idx,
            cat_dims=args.cat_dims,
            d_out=args.num_classes,
            task=args.objective,
        )

        self.to_device()

    # this is copied from TabSurvey's models.mlp.MLP
    def predict_helper(self, X):
        X = np.array(X, dtype=float)
        return super().predict_helper(X)

    @classmethod
    def define_trial_parameters(cls, trial, args):
        params = (
            {
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.0001, 0.01, log=True
                )
            },
        )
        return params

    @classmethod
    def get_random_parameters(cls, seed: int):
        rs = np.random.RandomState(seed)
        params = {"learning_rate": np.power(10, rs.uniform(-4, -2))}
        return params

    @classmethod
    def default_parameters(cls):
        params = {"learning_rate": 0.001}
        return params

    def fit(self, X, y, X_val=None, y_val=None):
        X = np.array(X, dtype=float)
        X_val = np.array(X_val, dtype=float)

        return super().fit(X, y, X_val, y_val)
