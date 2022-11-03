import os 
from pathlib import Path
import sys
# sys.path.insert(0, '/home/ramyasri/tabzilla/TabSurvey')
import tabzilla_preprocessors

# Import all preprocessor modules and add them to list for them to be in list of preprocessors
import tabzilla_preprocessors_openml

preprocessor_modules = [tabzilla_preprocessors, tabzilla_preprocessors_openml]

dataset_path = Path("datasets")


def build_preprocessors_dict():
    preprocessors = {}
    for module in preprocessor_modules:
        duplicates = preprocessors.keys() & module.preprocessor_dict.keys()
        if duplicates:
            raise RuntimeError(
                f"Duplicate dataset_name key found in module {module}: {duplicates}"
            )
        preprocessors.update(module.preprocessor_dict)
    return preprocessors


preprocessors = build_preprocessors_dict()



def main():
    # all_files = os.listdir("./../TabSurvey/datasets")
    # task_id = 168340
    # for file in all_files:
    #     words = file.split("__")
    #     if words[-1] == str(task_id):
    #         print(True)
    #         return

        
    # print(False)
    preprocessors = build_preprocessors_dict()
    for x in preprocessors:
        print(x)

if __name__ == "__main__":
    main()