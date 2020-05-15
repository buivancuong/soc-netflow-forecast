import pandas as pd
from soc_tsa.data_output.stat_model import pre_process, change_to_day_ts

training_data = pd.read_csv("./instance.csv")
training_data = change_to_day_ts(training_data)
training_data = pre_process(training_data)