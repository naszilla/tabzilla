from tabzilla_preprocessor_utils import dataset_preprocessor

preprocessor_dict = {}

# Now imported through OpenML
# @dataset_preprocessor(preprocessor_dict, "CaliforniaHousing", target_encode=False)
# def preprocess_cal_housing():
#     X, y = sklearn.datasets.fetch_california_housing(return_X_y=True)
#     return {
#         "X": X,
#         "y": y,
#         "cat_idx": [],
#         "target_type": "regression",
#         "num_classes": 1
#     }

# Now imported through OpenML
# @dataset_preprocessor(preprocessor_dict, "Covertype", target_encode=True)
# def preprocess_covertype():
#     X, y = sklearn.datasets.fetch_covtype(return_X_y=True)
#     return {
#         "X": X,
#         "y": y,
#         "cat_idx": [],
#         "target_type": "classification",
#         "num_classes": 7
#     }

# Now imported through OpenML
# @dataset_preprocessor(preprocessor_dict, "Adult", target_encode=True)
# def preprocess_adult():
#     url_data = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
#
#     features = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital-status', 'occupation',
#                 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
#     label = "income"
#     columns = features + [label]
#     df = pd.read_csv(url_data, names=columns)
#
#     df.fillna(0, inplace=True)
#
#     X = df[features].to_numpy()
#     y = df[label].to_numpy()
# #
#     return {
#         "X": X,
#         "y": y,
#         "cat_idx": [1,3,5,6,7,8,9,13],
#         "target_type": "binary",
#         "num_classes": 1
#     }