# list of algorithms + environments to be run with CPU. 
# these were run during our first set of experiments

# conda envs
SKLEARN_ENV="sklearn"
GBDT_ENV="gbdt"

MODELS_ENVS=(
    LinearModel:$SKLEARN_ENV
    KNN:$SKLEARN_ENV
    SVM:$SKLEARN_ENV
    DecisionTree:$SKLEARN_ENV
    RandomForest:$SKLEARN_ENV
    LightGBM:$GBDT_ENV
)
