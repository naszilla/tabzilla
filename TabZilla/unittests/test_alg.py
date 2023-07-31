# test the expeirment function with a user-specified model on five datasets
#
# usage: run from command line **without** the unittest command, with one positional argument.
# for example, to test the algorithm rtdl_MLP, run the following
#
# python -m unittests.test_alg rtdl_MLP
#
# *Do not* use the unittest command (for example: "python -m unittest unittests.test_alg <alg_name>"), since this will not pass the algorithm name as an argument

import argparse
import glob
import os
import shutil
import sys
import unittest
from unittest import TestCase

from unittests.test_experiments import TEST_DATASETS, test_experiment

RESULTS_DIR = "./unittests/results"


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

    def test_alg(self):
        for dataset, obj_type in TEST_DATASETS.items():
            with self.subTest(model=args.model_name, dataset=dataset):
                test_experiment(self, args.model_name, dataset, obj_type)
            # remove all contents from results dir
            self.cleanup_subtest()


# solution to pass args to unittest from here:
# https://stackoverflow.com/questions/44236745/parse-commandline-args-in-unittest-python
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(help="model name", dest="model_name")
    ns, args = parser.parse_known_args(namespace=unittest)
    return ns, sys.argv[:1] + args


if __name__ == "__main__":
    args, argv = parse_args()
    print(args, argv)
    sys.argv[:] = argv  # create cleans argv for main()
    unittest.main()
