# test the expeirment function with two models (LinearModel, RandomForest) on four datasets
# run using command line, from directory TabZilla:
#
# python -m unittest unittests.test_experiments
#
# OR
#
# python -m unittests.test_experiments


import glob
import json
import os
import shutil
import unittest
from unittest import TestCase

import tabzilla_experiment
from tabzilla_alg_handler import get_model
from tabzilla_utils import get_experiment_parser

TEST_DATASETS = {
    "openml__collins__3567": "classification",  # 15-class, 21 features 500 instances
    "openml__dermatology__35": "classification",  # 6-class, 34 features 366 instances
    "openml__credit-approval__29": "binary",  # binary, 15 features 690 instances
    "openml__sonar__39": "binary",  # binary, 60 features, 208 instances, with no cat features
    "openml__analcatdata_dmft__3560": "classification",  # classification, 4 features, no numerical features, 797 instances
}

CONFIG_FILE = "./unittests/test_config.yml"
DATASET_DIR = "./unittests/test_datasets"
RESULTS_DIR = "./unittests/results"


def test_experiment(self, model_name, dataset_name, obj_type):
    """run an experiment and check results"""

    # prepare experiment args from config file
    experiment_parser = get_experiment_parser()
    experiment_args = experiment_parser.parse_args(
        args="-experiment_config " + CONFIG_FILE
    )

    # try to get the model
    model_class = get_model(model_name)

    # if the objective type is not implemented, skip this test
    if obj_type in model_class.objtype_not_implemented:
        print(
            f"obj type {obj_type} not implemented for model {model_name}. skipping this test."
        )

    else:
        # run experiment
        tabzilla_experiment.main(
            experiment_args, model_name, DATASET_DIR + "/" + dataset_name
        )

        #### read results and run some sanity checks

        # make sure there are two results files
        result_files = glob.glob(RESULTS_DIR + "/*results.json")
        predictions_files = glob.glob(RESULTS_DIR + "/*predictions.json")
        self.assertEqual(len(result_files), 2)
        self.assertEqual(len(predictions_files), 2)

        # read the first result file
        with open(result_files[0], "r") as f:
            result = json.load(f)

        # make sure dataset and model name is correct
        self.assertEqual(dataset_name, result["dataset"]["name"])
        self.assertEqual(model_name, result["model"]["name"])

        ## check timer results
        self.assertIn("timers", result.keys())
        self.assertIn("train", result["timers"].keys())
        self.assertIn("val", result["timers"].keys())
        self.assertIn("test", result["timers"].keys())

        # get number of folds
        num_folds = len(result["timers"]["train"])
        self.assertEqual(num_folds, len(result["timers"]["val"]))
        self.assertEqual(num_folds, len(result["timers"]["test"]))

        # make sure training time is positive number for each fold
        self.assertTrue(all([t > 0 for t in result["timers"]["train"]]))

        ## check objective results
        self.assertIn("scorers", result.keys())
        self.assertIn("train", result["scorers"].keys())
        self.assertIn("val", result["scorers"].keys())
        self.assertIn("test", result["scorers"].keys())

        if obj_type == "classification":
            # classification objective: check log-loss
            metric = "Log Loss"
        else:
            # binary objective: check AUC
            metric = "AUC"

        self.assertIn(metric, result["scorers"]["val"].keys())
        self.assertIn(metric, result["scorers"]["test"].keys())
        self.assertEqual(num_folds, len(result["scorers"]["val"][metric]))
        self.assertEqual(num_folds, len(result["scorers"]["test"][metric]))
        self.assertTrue(
            all([isinstance(x, float) for x in result["scorers"]["test"][metric]])
        )


class TestExperiment(TestCase):
    @classmethod
    def setUp(cls):
        # create results folder. this is run before each test (not each subtest)
        shutil.rmtree(RESULTS_DIR, ignore_errors=True)
        os.mkdir(RESULTS_DIR)

    @classmethod
    def tearDown(cls):
        # remove results folder. this is run before each test (not each subtest)
        shutil.rmtree(RESULTS_DIR)

    @classmethod
    def cleanup_subtest(cls):
        # remove all contents from results folder
        for f in glob.glob(RESULTS_DIR + "/*"):
            os.remove(f)

    def test_linearmodel(self):
        # run subtest for this model
        model_name = "LinearModel"
        for dataset, obj_type in TEST_DATASETS.items():
            with self.subTest(model=model_name, dataset=dataset):
                test_experiment(self, model_name, dataset, obj_type)
            # remove all contents from results dir
            self.cleanup_subtest()

    def test_randomforest(self):
        # run subtest for this model
        model_name = "RandomForest"
        for dataset, obj_type in TEST_DATASETS.items():
            with self.subTest(model=model_name, dataset=dataset):
                test_experiment(self, model_name, dataset, obj_type)
            # remove all contents from results dir
            self.cleanup_subtest()


if __name__ == "__main__":
    unittest.main()
