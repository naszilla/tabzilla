import sqlite3
import pandas as pd
import optuna

# study_name = "LinearModel_CaliforniaHousing"
study_name = "KNN_Adult"

db_name = "sqlite:///{}.db".format(study_name)

study = optuna.load_study(study_name, db_name) 

df = study.trials_dataframe()
# with sqlite3.connect('/home/duncan/tabzilla/LinearModel_CaliforniaHousing.db') as db:
#     df = pd.read_sql_query("SELECT * FROM table_name", db_connection)