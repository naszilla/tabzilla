from typing import Any, List

import math
import numpy as np
import torch
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning import Trainer
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import pathlib
import pandas as pd 

from .hopular.blocks import Hopular
from .hopular.auxiliary.data import FixedDataset, DataModule
from models.basemodel_torch import BaseModelTorch

class Hopular_model(BaseModelTorch):
    """
    Model class for Hopular Networks for Tabular Data
    """

    BATCH_SIZE = 8

    def __init__(self, params, args) -> None:
        super().__init__(params, args)

        self.target_discrete = [args.num_features] if args.objective == "classification" else None
        self.target_numeric = [args.num_features] if args.objective == "regression" else None

        # trainer will be populated during fit()
        self.trainer = None

        if args.objective == "binary":
            self.target_discrete = [1]

        num_mask = np.ones(args.num_features)
        num_mask[args.cat_idx] = 0
        self.num_idx = np.where(num_mask)[0]
        self.model = None
    
    def write_dataframe(self, X, y, X_val=None, y_val=None):
        train_split = np.zeros(X.shape[0] + X_val.shape[0])
        train_split[:X.shape[0]] = 1
        val_split = 1 - train_split

        total_X_np_arr = pd.DataFrame(np.vstack((X, X_val)))
        total_y_np_arr = pd.DataFrame(np.concatenate((y, y_val)))

        # save to df
        resources_path = pathlib.Path(__file__).parent / r'hopular' / r'auxiliary' / r'resources' / self.args.dataset
        resources_path.mkdir(parents=True, exist_ok=True)
        total_X_np_arr.to_csv(resources_path / f'{self.args.dataset}_py.dat', index=False)
        total_X_np_arr.to_csv(resources_path / r'labels_py.dat', index=False)
        pd.DataFrame(train_split).to_csv(resources_path / r'folds_py.dat', index=False)
        pd.DataFrame(val_split).to_csv(resources_path / r'validation_folds_py.dat', index=False)

    def fit(self, X, y, X_val=None, y_val=None):
        self.write_dataframe(X, y, X_val, y_val)
        
        split_index = 0
        dataset = FixedDataset(
            dataset_name=self.args.dataset,
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
        max_epochs = 10 * 1  #temporarily shortened from 5000
        num_mini_batches = max(1, int(math.ceil(len(data_module.dataset.split_train) / self.BATCH_SIZE)))
        hopular_callback = ModelCheckpoint(monitor=r'hp_metric/val', mode=data_module.dataset.checkpoint_mode.value)
        self.trainer = Trainer(
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
        self.trainer.fit(model=self.model, datamodule=data_module)

        return None, None
    

    def predict_helper(self, X):

        X = np.array(X, dtype=np.float)
        X = torch.tensor(X).float()
        test_dataset = TensorDataset(X)
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=self.args.val_batch_size,
            shuffle=False,
            num_workers=2,
        )
        predictions = []
        with torch.no_grad():
            for batch_X in test_loader:
                preds = self.model(batch_X[0])

                if self.args.objective == "binary":
                    preds = torch.sigmoid(preds)

                predictions.append(preds) #.detach().cpu().numpy())
        return np.concatenate(predictions)


    @classmethod
    def define_trial_parameters(cls, trial, args):
        params = dict()
        return params

    @classmethod
    def get_random_parameters(cls, seed: int):
        rs = np.random.RandomState(seed)
        params = dict()
        return params

    @classmethod
    def default_parameters(cls):
        params = dict()
        return params







