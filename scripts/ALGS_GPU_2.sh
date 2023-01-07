# list of algorithms + environments to be run with GPU. 
# these were not included in the first set of experiments

# conda envs
TORCH_ENV="torch"
KERAS_ENV="tensorflow"
SKLEARN_ENV="sklearn"
GBDT_ENV="gbdt"

MODELS_ENVS=(
    TabTransformer:$TORCH_ENV  # Quite a few bugs here.
    NODE:$TORCH_ENV
    STG:$TORCH_ENV
    NAM:$TORCH_ENV  # Not implemented for classification
    DeepFM:$TORCH_ENV  # Not implemented for classification
    SAINT:$TORCH_ENV
    DANet:$TORCH_ENV
    rtdl_MLP:$TORCH_ENV
    rtdl_ResNet:$TORCH_ENV
)

# Not included:
# DeepGBM (bug)
# Hopular



