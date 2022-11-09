import time
import string
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from utils.io_utils import get_output_path

from models.basemodel import BaseModel


class BaseModelTorch(BaseModel):
    def __init__(self, params, args):
        super().__init__(params, args)
        self.device = self.get_device()
        self.gpus = (
            args.gpu_ids
            if args.use_gpu and torch.cuda.is_available() and args.data_parallel
            else None
        )

        # tabzilla: use a random string for temporary saving/loading of the model. pass this to load/save model functions
        self.tmp_name = "tmp_" + ''.join(random.sample(string.ascii_uppercase + string.digits, k=12))

    def to_device(self):
        if self.args.data_parallel:
            self.model = nn.DataParallel(self.model, device_ids=self.args.gpu_ids)

        print("On Device:", self.device)
        self.model.to(self.device)

    def get_device(self):
        if self.args.use_gpu and torch.cuda.is_available():
            if self.args.data_parallel:
                device = (
                    "cuda"  # + ''.join(str(i) + ',' for i in self.args.gpu_ids)[:-1]
                )
            else:
                device = "cuda"
        else:
            device = "cpu"

        return torch.device(device)

    # TabZilla: added a time limit
    def fit(self, X, y, X_val=None, y_val=None, time_limit=600):
        optimizer = optim.AdamW(
            self.model.parameters(), lr=self.params["learning_rate"]
        )

        X = torch.tensor(X).float()
        X_val = torch.tensor(X_val).float()

        y = torch.tensor(y)
        y_val = torch.tensor(y_val)

        if self.args.objective == "regression":
            loss_func = nn.MSELoss()
            y = y.float()
            y_val = y_val.float()
        elif self.args.objective == "classification":
            loss_func = nn.CrossEntropyLoss()
        else:
            loss_func = nn.BCEWithLogitsLoss()
            y = y.float()
            y_val = y_val.float()

        train_dataset = TensorDataset(X, y)
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=2,
        )

        val_dataset = TensorDataset(X_val, y_val)
        val_loader = DataLoader(
            dataset=val_dataset, batch_size=self.args.val_batch_size, shuffle=True
        )

        min_val_loss = float("inf")
        min_val_loss_idx = 0

        loss_history = []
        val_loss_history = []

        start_time = time.time()
        for epoch in range(self.args.epochs):
            for i, (batch_X, batch_y) in enumerate(train_loader):

                out = self.model(batch_X.to(self.device))

                if (
                    self.args.objective == "regression"
                    or self.args.objective == "binary"
                ):
                    # out = out.squeeze()
                    out = out.reshape((batch_X.shape[0], ))

                loss = loss_func(out, batch_y.to(self.device))
                loss_history.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Early Stopping
            val_loss = 0.0
            val_dim = 0
            for val_i, (batch_val_X, batch_val_y) in enumerate(val_loader):
                out = self.model(batch_val_X.to(self.device))

                if (
                    self.args.objective == "regression"
                    or self.args.objective == "binary"
                ):
                    #out = out.squeeze()
                    out = out.reshape((batch_val_X.shape[0], ))

                val_loss += loss_func(out, batch_val_y.to(self.device))
                val_dim += 1

            val_loss /= val_dim
            val_loss_history.append(val_loss.item())

            print("Epoch %d, Val Loss: %.5f" % (epoch, val_loss))

            if val_loss < min_val_loss:
                min_val_loss = val_loss
                min_val_loss_idx = epoch

                # Save the currently best model
                self.save_model(filename_extension="best", directory=self.tmp_name)

            if min_val_loss_idx + self.args.early_stopping_rounds < epoch:
                print(
                    "Validation loss has not improved for %d steps!"
                    % self.args.early_stopping_rounds
                )
                print("Early stopping applies.")
                break

            runtime = time.time() - start_time
            if runtime > time_limit:
                print(
                    f"Runtime has exceeded time limit of {time_limit} seconds. Stopping fit."
                )
                break

        # Load best model
        self.load_model(filename_extension="best", directory=self.tmp_name)
        return loss_history, val_loss_history

    def predict(self, X):
        # tabzilla update: return prediction probabilities
        if self.args.objective == "regression":
            self.predictions = self.predict_helper(X)
            probs = np.array([])
        else:
            self.predict_proba(X)
            self.predictions = np.argmax(self.prediction_probabilities, axis=1)
            probs = self.prediction_probabilities

        return self.predictions, probs

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probas = self.predict_helper(X)

        # If binary task returns only probability for the true class, adapt it to return (N x 2)
        if probas.shape[1] == 1:
            probas = np.concatenate((1 - probas, probas), 1)

        self.prediction_probabilities = probas
        return self.prediction_probabilities

    def predict_helper(self, X):
        self.model.eval()

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
                preds = self.model(batch_X[0].to(self.device))

                if self.args.objective == "binary":
                    preds = torch.sigmoid(preds)

                predictions.append(preds.detach().cpu().numpy())
        return np.concatenate(predictions)

    def save_model(self, filename_extension="", directory="models"):
        filename = get_output_path(
            self.args,
            directory=directory,
            filename="m",
            extension=filename_extension,
            file_type="pt",
        )
        torch.save(self.model.state_dict(), filename)

    def load_model(self, filename_extension="", directory="models"):
        filename = get_output_path(
            self.args,
            directory=directory,
            filename="m",
            extension=filename_extension,
            file_type="pt",
        )
        state_dict = torch.load(filename)
        self.model.load_state_dict(state_dict)

    def get_model_size(self):
        model_size = sum(t.numel() for t in self.model.parameters() if t.requires_grad)
        return model_size

    @classmethod
    def define_trial_parameters(cls, trial, args):
        raise NotImplementedError("This method has to be implemented by the sub class")
