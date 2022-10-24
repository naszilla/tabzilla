
# importing sys
import sys
sys.path.insert(0, '/home/ramyasri/tabzilla/TabSurvey')

from tabzilla_datasets import TabularDataset
from pathlib import Path


dataset_path = '/home/ramyasri/tabzilla/TabSurvey/datasets/openml__dionis__189355/'
p = Path(dataset_path).resolve()

dataset = TabularDataset.read(p)
print(dataset.target_type, dataset.num_features, dataset.num_classes, dataset.num_instances)
