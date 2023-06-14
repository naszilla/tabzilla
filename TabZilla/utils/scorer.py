from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    f1_score,
    log_loss,
    roc_auc_score,
)
import numpy as np


def get_scorer(args):
    if args.objective == "regression":
        return RegScorer()
    elif args.objective == "classification":
        return ClassScorer()
    elif args.objective == "binary":
        return BinScorer()
    else:
        raise NotImplementedError('No scorer for "' + args.objective + '" implemented')


class Scorer:

    """
    y_true: (n_samples,)
    y_prediction: (n_samples,) - predicted classes
    y_probabilities: (n_samples, n_classes) - probabilities of the classes (summing to 1)
    """

    def eval(self, y_true, y_prediction, y_probabilities):
        raise NotImplementedError("Has be implemented in the sub class")

    def get_results(self):
        raise NotImplementedError("Has be implemented in the sub class")

    def get_objective_result(self):
        raise NotImplementedError("Has be implemented in the sub class")


class RegScorer(Scorer):
    def __init__(self):
        self.mses = []
        self.r2s = []
        self.direction = "minimize"  # we want to minimize MSE

    # y_probabilities is None for Regression
    def eval(self, y_true, y_prediction, y_probabilities):
        mse = mean_squared_error(y_true, y_prediction)
        r2 = r2_score(y_true, y_prediction)

        self.mses.append(mse)
        self.r2s.append(r2)

        return {"MSE": mse, "R2": r2}

    def get_results(self):

        return {
            "MSE": self.mses,
            "R2": self.r2s,
        }

    def get_objective_result(self):
        return np.mean(self.mses)


class ClassScorer(Scorer):
    def __init__(self):
        self.loglosses = []
        self.aucs = []
        self.accs = []
        self.f1s = []
        self.direction = "minimize"  # we want to minimize log-loss

    def eval(self, y_true, y_prediction, y_probabilities, labels=None):
        logloss = log_loss(y_true, y_probabilities, labels=labels)
        # auc = roc_auc_score(y_true, y_probabilities, multi_class='ovr')
        auc = roc_auc_score(y_true, y_probabilities, labels=labels, multi_class="ovo", average="macro")

        acc = accuracy_score(y_true, y_prediction)
        f1 = f1_score(
            y_true, y_prediction, average="weighted"
        )  # use here macro or weighted?

        self.loglosses.append(logloss)
        self.aucs.append(auc)
        self.accs.append(acc)
        self.f1s.append(f1)

        return {"Log Loss": logloss, "AUC": auc, "Accuracy": acc, "F1 score": f1}

    def get_results(self):

        return {
            "Log Loss": self.loglosses,
            "AUC": self.aucs,
            "Accuracy": self.accs,
            "F1": self.f1s,
        }

    def get_objective_result(self):
        return np.mean(self.loglosses)


class BinScorer(Scorer):
    def __init__(self):
        self.loglosses = []
        self.aucs = []
        self.accs = []
        self.f1s = []
        self.direction = "maximize"  # we want to maximize AUC

    def eval(self, y_true, y_prediction, y_probabilities):
        logloss = log_loss(y_true, y_probabilities)
        auc = roc_auc_score(y_true, y_probabilities[:, 1])

        acc = accuracy_score(y_true, y_prediction)
        f1 = f1_score(
            y_true, y_prediction, average="micro"
        )  # use here macro or weighted?

        self.loglosses.append(logloss)
        self.aucs.append(auc)
        self.accs.append(acc)
        self.f1s.append(f1)

        return {"Log Loss": logloss, "AUC": auc, "Accuracy": acc, "F1 score": f1}

    def get_results(self):

        return {
            "Log Loss": self.loglosses,
            "AUC": self.aucs,
            "Accuracy": self.accs,
            "F1": self.f1s,
        }

    def get_objective_result(self):
        return np.mean(self.aucs)
