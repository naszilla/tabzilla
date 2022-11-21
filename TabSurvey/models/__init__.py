all_models = ["LinearModel", "KNN", "DecisionTree", "RandomForest", "XGBoost", "CatBoost", "LightGBM", "ModelTree",
               "MLP", "TabNet", "VIME", "TabTransformer", "NODE", "DeepGBM", "RLN", "DNFNet", "STG", "NAM", "DeepFM",
               "SAINT", "DANet"]


def str2model(model):

    if model == "LinearModel":
        from models.baseline_models import LinearModel
        return LinearModel

    elif model == "KNN":
        from models.baseline_models import KNN
        return KNN

    elif model == "SVM":
        from models.baseline_models import SVM
        return SVM

    elif model == "SVM_LINEAR":
        from models.baseline_models import SVM_LINEAR
        return SVM_LINEAR
    
    elif model == "SVM_SIGMOID":
        from models.baseline_models import SVM_SIGMOID
        return SVM_SIGMOID
    
    elif model == "SVM_POLY":
        from models.baseline_models import SVM_POLY
        return SVM_POLY

    elif model == "DecisionTree":
        from models.baseline_models import DecisionTree
        return DecisionTree

    elif model == "RandomForest":
        from models.baseline_models import RandomForest
        return RandomForest

    elif model == "XGBoost":
        from models.tree_models import XGBoost
        return XGBoost

    elif model == "CatBoost":
        from models.tree_models import CatBoost
        return CatBoost

    elif model == "LightGBM":
        from models.tree_models import LightGBM
        return LightGBM

    elif model == "MLP":
        from models.mlp import MLP
        return MLP

    elif model == "ModelTree":
        from models.modeltree import ModelTree
        return ModelTree

    elif model == "TabNet":
        from models.tabnet import TabNet
        return TabNet

    elif model == "VIME":
        from models.vime import VIME
        return VIME

    elif model == "TabTransformer":
        from models.tabtransformer import TabTransformer
        return TabTransformer

    elif model == "NODE":
        from models.node import NODE
        return NODE

    elif model == "DeepGBM":
        from models.deepgbm import DeepGBM
        return DeepGBM

    elif model == "RLN":
        from models.rln import RLN
        return RLN

    elif model == "DNFNet":
        from models.dnf import DNFNet
        return DNFNet

    elif model == "STG":
        from models.stochastic_gates import STG
        return STG

    elif model == "NAM":
        from models.neural_additive_models import NAM
        return NAM

    elif model == "DeepFM":
        from models.deepfm import DeepFM
        return DeepFM

    elif model == "SAINT":
        from models.saint import SAINT
        return SAINT

    elif model == "DANet":
        from models.danet import DANet
        return DANet

    else:
        raise NotImplementedError("Model \"" + model + "\" not yet implemented")
