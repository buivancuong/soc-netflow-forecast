'''
Ham chinh cua chuong trinh de chay chuong trinh voi input la file du lieu output la 144 diem du bao va do chinh xac cua mo hinh
'''
import time
import base64
import pickle

from data_ingestion import *
from train_model import *
from test_model import *
from accurate import *

# Main


def main():
    t0 = time.clock()
    path = './all_data_3m_19_9.xlsx'
    #path = './1m/'
    num = case()
    data, index = get_data(path, num)
    df, data = get_prdata(data, index)
    _count = 0
    choose = ['yes', 'no']
    while 1:
        if _count == 10:
            break
        _choose = input("Co train lai model khong?yes/no: ")
        if _choose == choose[0]:
            trent = SAX_get_trend(df)
            original = get_original(df, data)
            residual = get_resisual(trent, original)

            _evaluate_model = evaluate_models(trent, 4, 4)
            with open('./_evaluate_model.pickle', 'wb') as handle:
                pickle.dump(_evaluate_model, handle,
                            protocol=pickle.HIGHEST_PROTOCOL)

            one_step_for_trend = pred_trend(trent)

            X, y, test_reswavelet_scaled, scaler, residual_wavelet = pre_trainLSTM(
                residual)
            lstm_model = fit_lstm(X, y, 1, 100, 7)
            with open('./lstm_model.pickle', 'wb') as handle:
                pickle.dump(lstm_model, handle,
                            protocol=pickle.HIGHEST_PROTOCOL)

            one_step_for_residual = pred_testLSTM(
                test_reswavelet_scaled, lstm_model, scaler, residual_wavelet, residual)
            one_step_forecast_result = pred_test_original(
                one_step_for_trend, one_step_for_residual, original)
            evaluate_model(original, one_step_forecast_result)
            print("Forecast time: ", time.clock()-t0)
            break
        else:
            try:
                with open('./_evaluate_model.pickle', 'rb') as handle:
                    _evaluate_model = pickle.load(handle)
                with open('./lstm_model.pickle', 'rb') as handle:
                    lstm_model = pickle.load(handle)
            except:
                print("HAVE ERRORS")
            trent = SAX_get_trend(df)
            original = get_original(df, data)
            residual = get_resisual(trent, original)

            one_step_for_trend = pred_trend(trent)
            X, y, test_reswavelet_scaled, scaler, residual_wavelet = pre_trainLSTM(
                residual)

            one_step_for_residual = pred_testLSTM(
                test_reswavelet_scaled, lstm_model, scaler, residual_wavelet, residual)
            one_step_forecast_result = pred_test_original(
                one_step_for_trend, one_step_for_residual, original)
            evaluate_model(original, one_step_forecast_result)
            print("Forecast time: ", time.clock()-t0)
            break


if __name__ == '__main__':
    main()
