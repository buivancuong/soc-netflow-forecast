'''
fundametal packages: -pandas for data;

                    -numpy for calucation;
 
                    -dateutil.parser offers a generic date/time string parser 
                    -matplotlib, seaborn for visualize
                    -sklearn for preprocessing data and metrics
                    -scipy.stats for statistical
                    -statsmodels for ARIMA model
                    -time, datatime, calendar for time
                    -keras for LSTM model

get_data: du lieu dau vao the hien gia tri cua 5 bien inbyte, outbyte, inpkts, outpkts, count voi khoang cach 10 phut mot. Ta dinh dang du lieu ve dang bien nguyen va bien thoi gian se chuyen ve dinh dang thu. Sau do get_prdata se lay du lieu cac ngay lam viec loai bo cac ngay khong lam viec
'''
from datetime import datetime
import glob


import random
import calendar
from datetime import date
import statistics
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import scale
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

from statsmodels.tsa.stattools import adfuller


from dateutil.parser import parse
import time


import numpy as np

# import seaborn as sns
# sns.set()


# Lay data


def get_data(path, num):
    df = pd.read_excel(path, thousands=',')
    times = list(df["@timestamp per 10 minutes"])
    keys = list(df.keys())
    for key in keys[1:num]:

        count = [float(x) for x in list(df[key])]

        start_time = parse(times[0]).date()
        end_time = parse(times[len(times) - 1]).date()
        range_date = pd.date_range(start=start_time, end=end_time)
        range_time = pd.date_range(start="00:00", end="23:50", freq="10T")
        range_time = [x.time() for x in range_time]
        table_time = {}

        for date in range_date:
            table_time[str(date.date())] = [0] * len(range_time)

        index = [str(x) for x in range_time]
        data = pd.DataFrame(data=table_time, index=index)
        for i in range(len(count)):
            get_time = parse(times[i])
            data[str(get_time.date())][str(get_time.time())] = int(count[i])

        for date in data.keys():
            if list(data[date]).count(0) >= 1:
                data = data.drop([date], axis=1)
    return data, index

# lay data theo thu


