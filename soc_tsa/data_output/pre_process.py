import pandas as pd
import pywt
import sklearn.datasets
from sklearn.datasets.base import Bunch
from creme import preprocessing, stream
import creme
import numpy as np
from numpy import array, hstack
from statistics import mean
from datetime import datetime
from dateutil.parser import parse

from soc_tsa.utils.io_utils.data_utils import DataUtils


class PreProcess():
    def __init__(self):
        pass

    def data_ingestion(self, df):
        # all_data = pd.DataFrame()

        # for f in glob.glob(path + "*.csv"):
        #     df = pd.read_csv(f)
        #     all_data = all_data.append(df, ignore_index=True)

        # all_data.to_excel("all_data_1m_8_11.xlsx", index=False)
        # df = pd.read_excel("all_data_1m_8_11.xlsx",
        #                    thousands=',', dtype=object, freq='10min')
        ##################################
        times = [datetime.fromtimestamp(x / 1000.0)
                 for x in list(df["@timestamp per 10 minutes"])]
        times = [x.strftime('%Y-%m-%d %H:%M:%S') for x in times]

        count = [float(x) for x in list(df['value'])]

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
            if list(data[date]).count(0) >= 144:
                data = data.drop([date], axis=1)
        #data = data.replace(0,-9999)
        data = data.fillna(method='pad')
        for date in data.keys():
            if list(data[date]).count(0) >= 1:
                data = data.drop([date], axis=1)

        return data

    def trich_dl(self, df):
        A = []
        B = []
        C = []
        D = []
        E = []
        GTTB = []
        temp_table = pd.DataFrame([])
        for column in range(df.shape[1]):
            list_score = []
            for row in range(len(df)):
                value = df.iloc[row, column]
                q = df.iloc[row, :].quantile([.2, .4, .6, .8])
                level_1 = q.iloc[0].mean()
                level_2 = q.iloc[1].mean()
                level_3 = q.iloc[2].mean()
                level_4 = q.iloc[3].mean()

                score = ''
                if (value <= level_1):
                    score = 'A'
                    A.append(value)
                elif (value > level_1) & (value <= level_2):
                    score = 'B'
                    B.append(value)
                elif (value > level_2) & (value <= level_3):
                    score = 'C'
                    C.append(value)
                elif (value > level_3) & (value <= level_4):
                    score = 'D'
                    D.append(value)
                elif (value > level_4):
                    score = 'E'
                    E.append(value)
                list_score.append(score)
            temp_table[column + 1] = list_score
        GTTB = [A, B, C, D, E]
        temp_table.index = df.index
        return (temp_table, GTTB)

    def get_len_fit_wavelet(self, len_residual):
        if len_residual % 4 == 0:
            return len_residual
        else:
            return self.get_len_fit_wavelet(len_residual-1)

    # ap dung bien doi wavelet cho residual data
    def wavelet_transform(self, _residual_data):
        (ca, cd) = pywt.dwt(_residual_data, 'haar')  # Lọc lần 1
        (ca1, cd1) = pywt.dwt(cd, 'haar')  # Lọc lần 2 ; cần tìm hiểu haar
        # bỏ đi chuỗi cao lần 2 để lấy lại chuỗi residual_data_rec1
        residual_data_rec1 = pywt.idwt(ca1, None, 'haar')
        # gộp phần chuỗi thấp và residual_data_rec1
        residual_data_rec = pywt.idwt(ca, residual_data_rec1, 'haar')
        return residual_data_rec

    def timeseries_to_supervised(self, data, lag1=1, lag2=1):
        df = pd.DataFrame(data)
        columns1 = [df.shift(i) for i in range(1, lag1+1)]
        columns2 = [df.shift(i) for i in range(1, lag2+1)]
        X = pd.concat(columns1, axis=1)
        y = pd.concat(columns2, axis=1)
        return X.values, y.values

    # create a differenced series
    def difference(self, dataset, interval=1):
        diff = list()
        for i in range(interval, len(dataset)):
            value = dataset[i] - dataset[i - interval]
            diff.append(value)
        return pd.Series(diff)

    def scale(self, train, y_train):
        dataset = sklearn.datasets.base.Bunch(data=train, target=y_train)
        scaler = creme.preprocessing.MinMaxScaler()
        train_scaled = pd.DataFrame([])
        y_train_scaled = pd.DataFrame([])
        for xi, yi in stream.iter_sklearn_dataset(dataset):
            # Scale the features
            xi_scaled = scaler.fit_one(xi).transform_one(xi)
            # Scale target
            yi_scaled = scaler.fit_one(yi).transform_one(yi)
            train_scaled = train_scaled.append(
                pd.DataFrame(xi_scaled, index=[0]))
            y_train_scaled = y_train_scaled.append(
                pd.DataFrame(yi_scaled, index=[0]))
        return train_scaled.values, y_train_scaled.values

    def integrate(self, pre_y, yhat_arima, yhat_lstm):
        pred_sax_lstm = []
        for trend, residual, i in zip(yhat_arima, np.array(yhat_lstm).reshape(-1), range(pre_y.shape[0])):
            dataset = sklearn.datasets.base.Bunch(
                data=pre_y[:, i].reshape(pre_y.shape[0], 1), target=None)
            scaler = creme.preprocessing.MinMaxScaler()
            m = 0
            for xi, yi in stream.iter_sklearn_dataset(dataset):
                xi_scaled = scaler.fit_one(xi).transform_one(xi)
                if(m >= pre_y.shape[0]-1):
                    # Scale
                    current_min = scaler.min[0].get()
                    current_max = scaler.max[0].get()

                    pred_lstm = residual * \
                        (current_max - current_min) + current_min

                    # Diff
                    pred_lstm = pred_lstm + pre_y[-1][i]

                    #pre_y = np.append(pre_y, np.array(pred_lstm))
                    pred_sax_lstm.append(trend + pred_lstm)
                m = m + 1
        return pred_sax_lstm

    def fit_t(self, df, sensor, ts_field):
        # Chuỗi gốc
        original = []
        for i in range(0, df.shape[1]):
            for j in range(len(df.iloc[:, i])):
                original.append(df.iloc[j, i])
        # Tách df = SAX + Residual
        # 1) SAX
        b = self.trich_dl(df)
        data_symbol_full = b[0]
        GTTB = b[1]
        A = mean(GTTB[0])
        B = mean(GTTB[1])
        C = mean(GTTB[2])
        D = mean(GTTB[3])
        E = mean(GTTB[4])
        trend = []
        for i in range(df.shape[1]):
            for j in range(len(data_symbol_full.iloc[:, i])):
                value = data_symbol_full.iloc[j, i]
                if (value == 'A'):
                    trend.append(A)
                elif (value == 'B'):
                    trend.append(B)
                elif (value == 'C'):
                    trend.append(C)
                elif (value == 'D'):
                    trend.append(D)
                elif (value == 'E'):
                    trend.append(E)

        # 2) Residual
        residual = []
        for i in range(len(trend)):
            residual.append(abs(trend[i]-original[i]))

        residual_wavelet = self.wavelet_transform(
            residual[len(residual)-self.get_len_fit_wavelet(len(residual)):])
        diff_rw = self.difference(residual_wavelet, 1)
        # Input (X, y) - lstm
        lag1 = 80       # 80
        lag2 = 12        # 12
        X, y = self.timeseries_to_supervised(diff_rw, lag1, lag2)
        # Scale min-max (X, y)
        # X_scaled, y_scaled = self.scale(X, y)
        data_utils = DataUtils()
        X /= (data_utils.params['scaler'][sensor] *
              data_utils.params['scaler'][ts_field])
        y /= (data_utils.params['scaler'][sensor] *
              data_utils.params['scaler'][ts_field])    
        # Get X_test (-lag2:)
        X_test = X_scaled[-lag2:].copy()
        X_scaled = pd.DataFrame(X_scaled).iloc[lag1:-lag2, :].values
        y_scaled = pd.DataFrame(y_scaled).iloc[lag1+lag2:, :].values
        print(X[100])
        print(X_scaled[100])
        # print(len(X_scaled[100]))
        return (original, trend, residual, (X_scaled, y_scaled, X_test), y)


def split_sequence_stacked_lstm(data_sequence, memory_steps_in, predict_steps_out):
    X_feature = list()
    y_label = list()
    for i in range(len(data_sequence)):
        # find the end of this pattern
        end_in_index = i + memory_steps_in
        end_out_index = end_in_index + predict_steps_out
        # check if beyond the sequence
        if end_out_index > len(data_sequence):
            break
        # gather feature sequence and label sequence to the pattern
        feature_sequence = data_sequence[i:end_in_index]
        label_sequence = data_sequence[end_in_index:end_out_index]
        X_feature.append(feature_sequence)
        y_label.append(label_sequence)
    return array(X_feature), array(y_label)
