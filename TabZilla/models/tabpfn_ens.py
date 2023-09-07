from ensemble_tabpfn import EnsembleTabPFN
from models.basemodel import BaseModel

class TabPFNEnsModel(BaseModel):
    def __init__(self, params, args):
        super().__init__(params, args)

        if args.objective == "regression":
            raise NotImplementedError("Does not support")
        elif args.objective == "classification":
            self.model = EnsembleTabPFN(max_iters=50, n_ensemble_configurations=16)
        elif args.objective == "binary":
            self.model = EnsembleTabPFN(max_iters=50, n_ensemble_configurations=16)

    def fit(self, X, y, X_val=None, y_val=None):
        return super().fit(X, y)

    def predict_helper(self, X):
        return self.model.predict(X)

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

