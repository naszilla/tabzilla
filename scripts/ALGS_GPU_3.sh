# list of algorithms + environments to be run with GPU. 
# these were not included in the first set of experiments

# conda envs
TORCH_ENV="torch"
KERAS_ENV="tensorflow"
SKLEARN_ENV="sklearn"
GBDT_ENV="gbdt"

MODELS_ENVS=(
    rtdl_FTTransformer:$TORCH_ENV
    TabPFNModel:$TORCH_ENV
)




