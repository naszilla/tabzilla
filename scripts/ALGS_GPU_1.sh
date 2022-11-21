# list of algorithms + environments to be run with GPU. 
# these were run during our first set of experiments

# conda envs
TORCH_ENV="torch"
KERAS_ENV="tensorflow"
SKLEARN_ENV="sklearn"
GBDT_ENV="gbdt"

MODELS_ENVS=(
    XGBoost:$GBDT_ENV
    CatBoost:$GBDT_ENV
    MLP:$TORCH_ENV
    TabNet:$TORCH_ENV
    VIME:$TORCH_ENV
)


