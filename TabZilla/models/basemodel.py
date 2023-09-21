import typing as tp

import numpy as np
import optuna
import math
from utils.io_utils import save_model_to_file, save_predictions_to_file


class BaseModel:
    """Basic interface for all models.

    All implemented models should inherit from this base class to provide a common interface.
    At least they have to extend the init method defining the model and the define_trial_parameters method
    specifying the hyperparameters.

    Methods
    -------
    __init__(params, args):
        Defines the model architecture, depending on the hyperparameters (params) and command line arguments (args).
    fit(X, y, X_val=None, y_val=None)
        Trains the model on the trainings dataset (X, y). Validates the training process and uses early stopping
        if a validation set (X_val, y_val) is provided. Returns the loss history and validation loss history.
    predict(X)
        Predicts the labels of the test dataset (X). Saves and returns the predictions.
    attribute(X, y)
        Extract feature attributions for input pair (X, y)
    define_trial_parameters(trial, args)
        Returns a possible hyperparameter configuration. This method is necessary for the automated hyperparameter
        optimization.
    save_model_and_prediction(y_true, filename_extension="")
        Saves the current state of the model and the predictions and true labels of the test dataset.
    save_model(filename_extension="")
        Saves the current state of the model.
    save_predictions(y_true, filename_extension="")
        Saves the predictions and true labels of the test dataset.
    clone()
        Creates a fresh copy of the model using the same parameters, but ignoring any trained weights. This method
        is necessary for the cross validation.
    """

    # this list should be populated with "classification", "regression", and "binary" for each subclass if 
    # the model is not implemented for these objective types
    objtype_not_implemented = []

    def __init__(self, params: tp.Dict, args):
        """Defines the model architecture.

        After calling this method, self.model has to be defined.

        :param params: possible hyperparameter configuration, model architecture depends on this
        :param args: command line arguments containing all important information about the dataset and training process
        """
        self.args = args
        self.params = params

        # Model definition has to be implemented by the concrete model
        self.model = None

        # Create a placeholder for the predictions on the test dataset
        self.predictions = None
        self.prediction_probabilities = (
            None  # Only used by binary / multi-class-classification
        )

    # added for TabZilla bookkeeping
    def get_metadata(self):
        return {
            "name": self.__class__.__name__,
            "params": self.params,
            "args": self.args._asdict(),
        }

    def get_classes(self):
        if "classes_" not in dir(self):
            return None
        return self.classes_

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: tp.Union[None, np.ndarray] = None,
        y_val: tp.Union[None, np.ndarray] = None,
    ) -> tp.Tuple[list, list]:
        """Trains the model.

        The training is done on the trainings dataset (X, y). If a validation set (X_val, y_val) is provided,
        the model state is validated during the training, to allow early stopping.

        Returns the loss history and validation loss history if the loss and validation loss development during
        the training are logged. Otherwise empty lists are returned.

        :param X: trainings data
        :param y: labels of trainings data
        :param X_val: validation data
        :param y_val: labels of validation data
        :return: loss history, validation loss history
        """

        self.model.fit(X, y)

        # Should return loss history and validation loss history
        return [], []

    # Patch around the original predict method to handle case of missing classes in training set and subsampling.
    # This needs to be done as a separate method, since several of the inheriting classes override the predict_proba or
    # predict methods.
    def predict_wrapper(self, X: np.ndarray, max_rows : int) -> tp.Tuple[np.ndarray, np.ndarray]:
        if max_rows > 0 and X.shape[0] > max_rows:
            X_ens = []
            X_preds = []
            X_probas = []
            for idx, i in enumerate(range(0, X.shape[0], max_rows)):
                print(f"Fitting samples {idx+1} of {math.ceil(X.shape[0]/max_rows)}")
                X_ens.append(X[i:i+max_rows])
                preds, probas = self.predict(X_ens[-1])
                X_preds.append(preds)
                X_probas.append(probas)
            self.predictions, self.prediction_probabilities = np.concatenate(X_preds, axis=0), np.concatenate(X_probas, axis=0)
        else:
            self.predictions, self.prediction_probabilities = self.predict(X)
        if (
            self.args.objective == "classification"
            and self.prediction_probabilities.shape[1] != self.args.num_classes
        ):
            # Handle special case of missing classes in training set, which can (depending on the model)  result in
            # predictions only being made for those classes
            classes_ = self.get_classes()
            if classes_ is None:
                raise NotImplementedError(
                    f"Cannot infer classes for model of type {type(self)}"
                )
            # From https://github.com/scikit-learn/scikit-learn/issues/21568#issuecomment-984030911
            y_score_expanded = np.zeros(
                (self.prediction_probabilities.shape[0], self.args.num_classes),
                dtype=self.prediction_probabilities.dtype,
            )
            for idx, class_id in enumerate(classes_):
                y_score_expanded[:, class_id] = self.prediction_probabilities[:, idx]
            self.prediction_probabilities = y_score_expanded
            self.predictions = np.argmax(self.prediction_probabilities, axis=1)

        return self.predictions, self.prediction_probabilities

    def predict(self, X: np.ndarray) -> tp.Tuple[np.ndarray, np.ndarray]:
        """
        Returns the regression value or the concrete classes of binary / multi-class-classification tasks.
        (Save predictions to self.predictions)

        :param X: test data
        :return: predicted values / classes of test data (Shape N x 1)
        """

        # TabZilla update: always return prediction probabilities
        self.prediction_probabilities = np.array([])

        if self.args.objective == "regression":
            self.predictions = self.model.predict(X)
        elif self.args.objective == "classification" or self.args.objective == "binary":
            self.prediction_probabilities = self.predict_proba(X)
            self.predictions = np.argmax(self.prediction_probabilities, axis=1)

        return self.predictions, self.prediction_probabilities

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Only implemented for binary / multi-class-classification tasks.
        Returns the probability distribution over the classes C.
        (Save probabilities to self.prediction_probabilities)

        :param X: test data
        :return: probabilities for the classes (Shape N x C)
        """

        self.prediction_probabilities = self.model.predict_proba(X)

        # If binary task returns only probability for the true class, adapt it to return (N x 2)
        if self.prediction_probabilities.shape[1] == 1:
            self.prediction_probabilities = np.concatenate(
                (1 - self.prediction_probabilities, self.prediction_probabilities), 1
            )
        return self.prediction_probabilities

    def save_model_and_predictions(self, y_true: np.ndarray, filename_extension=""):
        """Saves the current state of the model and the predictions and true labels of the test dataset.

        :param y_true: true labels of the test data
        :param filename_extension: (optional) additions to the filenames
        """
        self.save_predictions(y_true, filename_extension)
        self.save_model(filename_extension)

    def clone(self):
        """Clone the model.

        Creates a fresh copy of the model using the same parameters, but ignoring any trained weights. This method
        is necessary for the cross validation.

        :return: Copy of the current model without trained parameters
        """
        return self.__class__(self.params, self.args)

    @classmethod
    def define_trial_parameters(cls, trial: optuna.Trial, args) -> tp.Dict:
        """Define the ranges of the hyperparameters

        Returns a possible hyperparameter configuration. This method is necessary for the automated hyperparameter
        optimization. All hyperparameter that should be optimized and their ranges are specified here.
        For more information see: https://optuna.org/

        :param trial: Trial class instance generated by the optuna library.
        :param args: Command line arguments containing all important information about the dataset
        :return: Hyperparameter configuration
        """

        raise NotImplementedError("This method has to be implemented by the sub class")

    # TabZilla: add placeholder methods for get_random_parameters() and default_parameters()
    @classmethod
    def get_random_parameters(cls, seed: int):
        """
        returns a random set of hyperparameters, which can be replicated using the provided seed
        """
        raise NotImplementedError("This method has to be implemented by the sub class")

    @classmethod
    def default_parameters(cls):
        """
        returns the default set of hyperparameters
        """
        raise NotImplementedError("This method has to be implemented by the sub class")

    def save_model(self, filename_extension=""):
        """Saves the current state of the model.

        Saves the model using pickle. Override this method if model should be saved in a different format.

        :param filename_extension: true labels of the test data
        """
        save_model_to_file(self.model, self.args, filename_extension)

    def save_predictions(self, y_true: np.ndarray, filename_extension=""):
        """Saves the predictions and true labels of the test dataset.

        Saves the predictions and the truth values together in a npy file.

        :param y_true: true labels of the test data
        :param filename_extension: true labels of the test data
        """
        if self.args.objective == "regression":
            # Save array where [:,0] is the truth and [:,1] the prediction
            y = np.concatenate(
                (y_true.reshape(-1, 1), self.predictions.reshape(-1, 1)), axis=1
            )
        else:
            # Save array where [:,0] is the truth and [:,1:] are the prediction probabilities
            y = np.concatenate(
                (y_true.reshape(-1, 1), self.prediction_probabilities), axis=1
            )

        save_predictions_to_file(y, self.args, filename_extension)

    def get_model_size(self):
        raise NotImplementedError(
            "Calculation of model size has not been implemented for this model."
        )

    def attribute(cls, X: np.ndarray, y: np.ndarray, strategy: str = "") -> np.ndarray:
        """Get feature attributions for inherently interpretable models. This function is only implemented for
        interpretable models.

        :param X: data (Shape N x D)
        :param y: labels (Shape N) for which the attribution should be computed for (
        usage of these labels depends on the specific model)

        :strategy: if there are different strategies that can be used to compute the attributions they can be passed
        here. Passing an empty sting should always result in the default strategy.

        :return The (non-normalized) importance attributions for each feature in each data point. (Shape N x D)
        """
        raise NotImplementedError(
            f"This method is not implemented for class {type(cls)}."
        )
