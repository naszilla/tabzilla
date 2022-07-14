# Overview

...


## Running Tabzilla Experiments

We modified the TabSurvey code in order to run experiments to generate results for our meta-learning tasks. The script [`TabSurvey/tabzilla_experiment.py`](TabSurvey/tabzilla_experiment.py) runs these experiments (this is adapted from the script [`TabSurvey/train.py`](TabSurvey/train.py)).

Similar to `test.py`, this script writes a database of various results from each train/test cycle. These results are written by optuna.

### `TabSurvey/tabzilla_experiment.py`

This script 