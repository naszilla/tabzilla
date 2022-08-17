import sqlite3
import pandas as pd
import optuna

study_name = "KNN_openml__california__361089"
# study_name = "LinearModel_openml__california__361089"

db_name = "sqlite:///{}.db".format(study_name)

study = optuna.load_study(study_name, db_name)

df = study.trials_dataframe()

# get summaries

# with sqlite3.connect('/home/duncan/tabzilla/LinearModel_CaliforniaHousing.db') as db:
#     df = pd.read_sql_query("SELECT * FROM table_name", db_connection)
