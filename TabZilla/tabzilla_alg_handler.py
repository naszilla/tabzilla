# this script defines two objects for accessing ML models/algorithms:
# - dictionary ALL_MODELS: each key is a model/alg name, and each value is a function that imports and returns the model class
# - function get_model(model_name), which returns the model class by evaluating the model-getter
#
# to add a new model/algorithm, simply add a new model-getter function that imports and returns the model class,
# and add the decorator @register_model to this function.

# dictionary of all model names
ALL_MODELS = {}


def register_model(func):
    """add model to the list of all models"""
    ALL_MODELS[func.__name__] = func
    return func


##############################################################
# sklearn-based models


@register_model
def LinearModel():
    from models.baseline_models import LinearModel as model

    return model


@register_model
def KNN():
    from models.baseline_models import KNN as model

    return model


@register_model
def SVM():
    from models.baseline_models import SVM as model

    return model


@register_model
def DecisionTree():
    from models.baseline_models import DecisionTree as model

    return model


@register_model
def RandomForest():
    from models.baseline_models import RandomForest as model

    return model


##############################################################
# gbdt models


@register_model
def XGBoost():
    from models.tree_models import XGBoost as model

    return model


@register_model
def CatBoost():
    from models.tree_models import CatBoost as model

    return model


@register_model
def LightGBM():
    from models.tree_models import LightGBM as model

    return model


# Not tested
# @register_model
# def ModelTree():
#     from models.modeltree import ModelTree as model

#     return model


##############################################################
# torch-based models


@register_model
def MLP():
    from models.mlp import MLP as model

    return model


@register_model
def TabNet():
    from models.tabnet import TabNet as model

    return model


@register_model
def VIME():
    from models.vime import VIME as model

    return model


@register_model
def TabTransformer():
    from models.tabtransformer import TabTransformer as model

    return model


@register_model
def NODE():
    from models.node import NODE as model

    return model


@register_model
def DeepGBM():
    from models.deepgbm import DeepGBM as model

    return model


@register_model
def STG():
    from models.stochastic_gates import STG as model

    return model


@register_model
def NAM():
    from models.neural_additive_models import NAM as model

    return model


@register_model
def DeepFM():
    from models.deepfm import DeepFM as model

    return model


@register_model
def SAINT():
    from models.saint import SAINT as model

    return model


@register_model
def DANet():
    from models.danet import DANet as model

    return model


# not implemented yet.
# @register_model
# def Hopular_model():
#     from models.hopular_model import Hopular_model as model

#     return model


@register_model
def TabPFNModel():
    from models.tabpfn import TabPFNModel as model

    return model


##############################################################
# rtdl models (also using torch)
# code: https://yura52.github.io
# paper: https://arxiv.org/abs/2106.11959


@register_model
def rtdl_MLP():
    from models.rtdl import rtdl_MLP as model

    return model


@register_model
def rtdl_ResNet():
    from models.rtdl import rtdl_ResNet as model

    return model


@register_model
def rtdl_FTTransformer():
    from models.rtdl import rtdl_FTTransformer as model

    return model


def get_model(model_name):
    if model_name in ALL_MODELS:
        # get the model-getter
        model_getter = ALL_MODELS[model_name]

        # evaluate the model-getting function to return the model class
        return model_getter()
    else:
        raise NotImplementedError(f"Model {model_name} not implemented")


if __name__ == "__main__":
    print("all algorithms:")
    for n in ALL_MODELS.keys():
        print(n)
