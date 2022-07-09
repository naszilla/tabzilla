from pathlib import Path
import json
from optuna.trial import FrozenTrial

def write_dict_to_json(x: dict, filepath: Path):
    assert not filepath.is_file(), f"file already exists: {filepath}"
    assert filepath.parent.is_dir(), f"directory does not exist: {filepath.parent}"
    with filepath.open("w", encoding="UTF-8") as f: 
        json.dump(x, f)

def trial_to_dict(trial):
    """return a dict representation of an optuna FrozenTrial"""
    assert isinstance(trial, FrozenTrial), f"trial must be of type optuna.trial.FrozenTrial. this object has type {type(trial)}"
    
    # get all user_metrics
    trial_dict = trial.user_attrs.copy()

    # add trial number
    trial_dict["trial_number"] = trial.number
    
    # add trial number
    trial_dict["trial_params_obj"] = trial.params
    
    # add system attributes
    trial_dict["system_attributes"] = trial.system_attrs

    return trial_dict

def write_trial_to_json(trial, filepath: Path):
    """write the dict representation of an optuna trial to file"""
    write_dict_to_json(trial_to_dict(trial), filepath)

