# adapted from models/__init__.py

import os

# dictionary of all model names, with the conda env they need to be run with
ALL_MODELS = {}


def register_model(conda_env):
    """
    decorator to store the name of all model functions in a dict with the name of the conda env they should be run with

    this decorator should be added to any function that gets a model class.

    the name of each model-getting function should have the format "get_<model-name>"

    we do this so that the model functions don't have to be imported unless they are needed
    """

    def decorator(func):
        ALL_MODELS[func.__name__[4:]] = (
            conda_env,
            func,
        )  # remove the "get" from the model name with [4:]
        return func

    return decorator


##############################################################
# sklearn models


@register_model("sklearn")
def get_LinearModel():
    from models.baseline_models import LinearModel as model

    return model


@register_model("sklearn")
def get_KNN():
    from models.baseline_models import KNN as model

    return model


@register_model("sklearn")
def get_SVM():
    from models.baseline_models import SVM as model

    return model


@register_model("sklearn")
def get_DecisionTree():
    from models.baseline_models import DecisionTree as model

    return model


@register_model("sklearn")
def get_RandomForest():
    from models.baseline_models import RandomForest as model

    return model


##############################################################
# gbdt models


@register_model("gbdt")
def get_XGBoost():
    from models.tree_models import XGBoost as model

    return model


@register_model("gbdt")
def get_CatBoost():
    from models.tree_models import CatBoost as model

    return model


@register_model("gbdt")
def get_LightGBM():
    from models.tree_models import LightGBM as model

    return model


@register_model("gbdt")
def get_ModelTree():
    from models.modeltree import ModelTree as model

    return model


##############################################################
# torch models


@register_model("torch")
def get_MLP():
    from models.mlp import MLP as model

    return model


@register_model("torch")
def get_TabNet():
    from models.tabnet import TabNet as model

    return model


@register_model("torch")
def get_VIME():
    from models.vime import VIME as model

    return model


@register_model("torch")
def get_TabTransformer():
    from models.tabtransformer import TabTransformer as model

    return model


@register_model("torch")
def get_NODE():
    from models.node import NODE as model

    return model


@register_model("torch")
def get_DeepGBM():
    from models.deepgbm import DeepGBM as model

    return model


@register_model("torch")
def get_STG():
    from models.stochastic_gates import STG as model

    return model


@register_model("torch")
def get_NAM():
    from models.neural_additive_models import NAM as model

    return model


@register_model("torch")
def get_DeepFM():
    from models.deepfm import DeepFM as model

    return model


@register_model("torch")
def get_SAINT():
    from models.saint import SAINT as model

    return model


@register_model("torch")
def get_DANet():
    from models.danet import DANet as model

    return model


##############################################################
# tensorflow models


@register_model("tensorflow")
def get_RLN():
    from models.rln import RLN as model

    return model


@register_model("tensorflow")
def get_DNFNet():
    from models.dnf import DNFNet as model

    return model


##############################################################
# rtdl models (using torch environment)
# code: https://yura52.github.io
# paper: https://arxiv.org/abs/2106.11959


@register_model("torch")
def get_rtdl_MLP():
    from models.rtdl import rtdl_MLP as model

    return model


@register_model("torch")
def get_rtdl_ResNet():
    from models.rtdl import rtdl_ResNet as model

    return model


@register_model("torch")
def get_rtdl_FTTransformer():
    from models.rtdl import rtdl_FTTransformer as model

    return model


def get_model(model_name):
    if model_name in ALL_MODELS:

        # get the model-getter and conda env
        conda_env, model_getter = ALL_MODELS[model_name]

        # make sure we are using the correct environment
        assert (
            os.environ["CONDA_DEFAULT_ENV"] == conda_env
        ), f"model {model_name} requires conda env {conda_env}. current env is: {os.environ['CONDA_DEFAULT_ENV']}"

        # evaluate the model-getting function to return the model class
        return model_getter()
    else:
        raise NotImplementedError(f"Model {model_name} not implemented")
