from typing import Any, List

import math
import numpy as np
import torch
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning import Trainer
import torch.nn as nn
import pathlib
import pandas as pd 

from .hopular.blocks import Hopular
from .hopular.auxiliary.data import FixedDataset, DataModule
from models.basemodel_torch import BaseModelTorch

class HopularModel(BaseModelTorch):
    """
    Model class for Hopular Networks for Tabular Data
    """

    BATCH_SIZE = 8

    def __init__(self, params, args) -> None:
        super().__init__(params, args)

        self.target_discrete = args.num_features if args.objective == "classification" else None
        self.target_numeric = args.num_features if args.objective == "regression" else None

        num_mask = np.ones(args.num_features)
        num_mask[args.cat_idx] = 0
        self.num_idx = np.where(num_mask)[0]
        self.model = None
    
    def write_dataframe(self, X, y, X_val=None, y_val=None):
        train_split = np.ones(X.shape[0]) + np.zeros(X_val.shape[0])
        val_split = 1 - train_split

        total_X_np_arr = pd.DataFrame(np.vstack((X, X_val)))
        total_y_np_arr = pd.DataFrame(np.vstack((y, y_val)))

        # save to df
        resources_path = pathlib.Path(__file__).parent / r'resources' / self.args.dataset_name
        total_X_np_arr.to_csv(resources_path / f'{self.args.dataset_name}_py.dat', index=False) # TODO: @duncan is this the right way to fetch the dataset name?
        total_X_np_arr.to_csv(resources_path / r'labels_py.dat', index=False) # TODO: @duncan is this the right way to fetch the dataset name
        pd.DataFrame(train_split).to_csv(resources_path / r'folds_py.dat', index=False)
        pd.DataFrame(val_split).to_csv(resources_path / r'validation_folds_py.dat', index=False)

    def fit(self, X, y, X_val=None, y_val=None):
        self.write_dataframe(X, y, X_val, y_val)
        
        split_index = 0
        dataset = FixedDataset(
            dataset_name=self.args.dataset_name,
            feature_numeric=self.num_idx,
            feature_discrete=self.args.cat_idx,
            target_discrete=self.target_discrete,
            target_numeric=self.target_numeric,
            missing_entries=None,
            split_index=split_index,
            split_state=1
        )

        data_module = DataModule(
            dataset=dataset,
            batch_size=self.BATCH_SIZE
        )

        self.model = Hopular.from_data_module(data_module=data_module)

        self.model.reset_parameters()

        # prepare trainer
        max_epochs = 5000 * 1
        num_mini_batches = max(1, int(math.ceil(len(data_module.dataset.split_train) / self.BATCH_SIZE)))
        hopular_callback = ModelCheckpoint(monitor=r'hp_metric/val', mode=data_module.dataset.checkpoint_mode.value)
        hopular_trainer = Trainer(
            max_epochs=max_epochs,
            log_every_n_steps=1,
            check_val_every_n_epoch=10,
            gradient_clip_val=1.0,
            gradient_clip_algorithm=r'norm',
            gpus=1 if torch.cuda.is_available() else 0,
            callbacks=[hopular_callback],
            deterministic=False,
            accumulate_grad_batches=num_mini_batches
        )

        # Fit and test Hopular instance (testing is done on chosen best model).
        hopular_trainer.fit(model=self.model, datamodule=data_module)
        return
    
    # this is copied from TabSurvey's models.mlp.MLP
    def predict_helper(self, X):
        X = np.array(X, dtype=np.float)
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







