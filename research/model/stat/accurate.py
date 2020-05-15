'''
Danh gia model su dung 3 chi so danh gia la MAPE, MAE va RMSE. Danh gia giua 144 diem du bao so voi 144 diem da co trong bo du lieu.
Input: 144 diem cua tuan cuoi cung cua bo du lieu tuong ung voi thu 2 va 144 diem du bao tuong ung.
Output: do chinh xac cua mo hinh du bao 
'''
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import mean_absolute_error
from test_model import *

# Đánh giá model


def Mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def Mean_absolute_error(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)


def Root_mean_squared_error(y_true, y_pred):
    return np.sqrt(((y_pred - y_true) ** 2).mean())


def evaluate_model(original, one_step_forecast_result):
    mape = Mean_absolute_percentage_error(
        original[-144:], one_step_forecast_result)
    mae = Mean_absolute_error(original[-144:], one_step_forecast_result)
    rmse = Root_mean_squared_error(original[-144:], one_step_forecast_result)

    print('Timeseries forecast: ', one_step_forecast_result)
    print('Mean absolute percentage error: ', mape)
    print('Mean absolute error: ', mae)
    print('Root mean squared error: ', rmse)
    return mape, mae, rmse


def case():
    count = 0
    possible_case = ['inbyte', 'outbyte', 'inpkts', 'outpkts', 'count']
    num1 = possible_case.index('inbyte') + 2
    num2 = possible_case.index('outbyte') + 2
    num3 = possible_case.index('inpkts') + 2
    num4 = possible_case.index('outpkts') + 2
    num5 = possible_case.index('count') + 2
    while 1:
        if count == 10:
            break
        num = input(
            "Nhap truong can du doan (inbyte/outbyte/inpkts/outpkts/count): ")
        if num == possible_case[0]:
            num = num1
            break
        if num == possible_case[1]:
            num = num2
            break
        if num == possible_case[2]:
            num = num3
            break
        if num == possible_case[3]:
            num = num4
            break
        if num == possible_case[4]:
            num = num5
            break
        else:
            print("Truong nhap khong hop le")
        count += 1
    return num
