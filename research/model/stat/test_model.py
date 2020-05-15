'''
Du bao phan test cua chuoi goc voi 144 diem.
Input la 2 bien cua file train_model la model_fit, lstm_model.
Output la chuoi thoi gian duoc du bao cua 144 diem.
(Output duoc the hien la bien one_step_forecast_result)
'''
import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

from train_model import *

# D? b�o cho chu?i g?c

# D? do�n m?t bu?c cho ph?n trend


def forecast_one_step_for_trend(_history, _test, order):
    history = [x for x in _history]
    predictions = list()
    # order =(8,1,7)
    # order = evaluate_models(history,10,10)
    # print(order)
    for i in range(len(_test)):
        model_fit = fit_model(history, order[0], order[1], order[2])
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(_test[i])
    return predictions


# D? do�n trend
def pred_trend(trent):
    one_step_for_trend = forecast_one_step_for_trend(
        trent[0:-144], trent[-144:], evaluate_models(trent, 4, 4))

    # V? trend d? b�o - trend test
    plt.figure(figsize=(20, 6))
    plt.plot(one_step_for_trend, label='trend - predict')
    plt.plot(trent[-144:], label='trend')
    plt.legend()
    plt.show()
    # plt.savefig('trend')
    return one_step_for_trend


# D? do�n 1 bu?c gi� tr? residual b?i LSTM
def forecast_one_step_for_lstm(model, X_, _scaler, _orgin_bfdiff):
    predictions = list()
    for i in range(len(X_)):
        X = X_[i]
        yhat = model.predict(X.reshape(1, 1, X.shape[1]))
        # invert scaling
        yhat = invert_scale(_scaler, X, yhat)
        # invert diffence
        yhat = inverse_difference(_orgin_bfdiff, yhat, len(X_) + 1 - i)
        # append predict
        predictions.append(yhat)
    return predictions


#  D? do�n Test ph?n residual
def pred_testLSTM(test_reswavelet_scaled, lstm_model, scaler, residual_wavelet, residual):
    X_, y_ = define_input(test_reswavelet_scaled)
    # print('X: ', X)
    # print('y: ', y)
    one_step_for_residual = forecast_one_step_for_lstm(
        lstm_model, X_, scaler, residual_wavelet)

    # V? d? do�n chu?i v� test
    plt.figure(figsize=(20, 6))
    plt.plot(one_step_for_residual, label='predict')
    plt.plot(residual[-144:], label='residual-original')
    plt.legend()
    plt.show()
    # plt.savefig('predict')
    return one_step_for_residual


def pred_test_original(one_step_for_trend, one_step_for_residual, original):
    one_step_forecast_result = np.asarray(
        [float(i) for i in one_step_for_trend]) + np.asarray(one_step_for_residual)

    # V? d? b�o - chu?i g?c test
    plt.figure(figsize=(20, 6))
    plt.plot(one_step_forecast_result, label='predict-result')
    plt.plot(original[-144:], label='original')
    plt.legend()
    plt.show()
    # plt.savefig('original.png')
    return one_step_forecast_result
