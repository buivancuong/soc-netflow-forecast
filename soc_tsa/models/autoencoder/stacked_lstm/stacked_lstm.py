from soc_tsa.utils.io_utils.data_utils import DataUtils
from soc_tsa.utils.io_utils.sqlite_access import SQLiteDB

from keras.models import Sequential
from keras.models import load_model
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.utils import multi_gpu_model

from numpy import array
import pandas as pd
import time
import datetime
import logging
import hashlib
import random


class AE_LSTM:

    def __init__(self, sensor, algorithm_model, ts_field):
        data_utils = DataUtils()
        self.sensor = sensor
        self.algorithm_model = algorithm_model
        self.ts_field = ts_field
        self.model_id = self.sensor + '_' + self.algorithm_model + '_' + self.ts_field
        self.model_path = data_utils.models_dir + self.model_id + '.h5'
        try:
            self.model = load_model(self.model_path)
            logging.warning("Load the existed model")
        except:
            logging.error(
                "Cannot find the model file in the directory, create new model")
            self.model = create_stacked_lstm_model(
                memory_steps_in=144, predict_steps_out=12)
            self.model.save(self.model_path)
            sqlite_db = SQLiteDB(data_utils.sqlite_dir)
            sqlite_db.create_db_connection()
            sqlite_db.add_model('models', self.model_id,
                                self.model_path, self.sensor, self.ts_field)

    def training_model(self, training_data, epochs=100):
        try:
            self.model = multi_gpu_model(model=self.model_id, gpus=2)
            self.model.compile(optimizer='adam', loss='mse')
        except Exception as err:
            print("**********************AE-LSTM*************************")
            print(err)
            print("**********************AE-LSTM*************************")
        memory_steps_in = self.model.layers[0].input_shape[1]
        predict_steps_out = self.model.layers[3].output_shape[1]
        X_feature, y_label = split_sequence_stacked_lstm(data_sequence=training_data.tolist(
        ), memory_steps_in=memory_steps_in, predict_steps_out=predict_steps_out)
        X_feature = X_feature.reshape(X_feature.shape[0], X_feature.shape[1], 1)
        y_label = y_label.reshape(y_label.shape[0], y_label.shape[1], 1)
        self.model.fit(x=X_feature, y=y_label, epochs=epochs, batch_size=32)
        self.model.save(self.model_path)

    def prediction(self, data_input, hours_predict=1):
        X_feature = data_input
        data_utils = DataUtils()
        ts_field_scale = data_utils.params['scaler']['ts_field']
        sensor_scale = data_utils.params['scaler']['sensor']
        memory_steps_in = int(self.model.input_shape[1])
        predict_steps_out = int(self.model.layers[1].output_shape[1])
        X_feature_curr = X_feature.reshape(1, memory_steps_in, 1)
        if int(X_feature_curr.shape[1]) < memory_steps_in:
            logging.error('Necessary length input is not enough!')
            return None
        oracle_sequence = list()
        if int(hours_predict) > 24:
            logging.warning("Bo chi cho doan 24h")
            hours_predict = 24

        for i in range(int(hours_predict)):
            y_hat = self.model.predict(X_feature_curr[-memory_steps_in:])
            oracle_sequence += y_hat.reshape(y_hat.shape[1]).tolist()
            # X_feature_curr = X_feature_curr.tolist()
            X_feature_curr = X_feature_curr.reshape(
                X_feature_curr.shape[1]).tolist() + oracle_sequence[-predict_steps_out:]
            X_feature_curr = array(
                X_feature_curr[-memory_steps_in:]).reshape(1, memory_steps_in, 1)

        data_utils = DataUtils()
        oracle_seq_doc = list()
        start_timestamp = int(time.time()) - (int(time.time() % 300))
        for value in oracle_sequence:
            value_doc = {}
            value_doc['value'] = value * \
                ts_field_scale[self.ts_field] * sensor_scale[self.sensor]
            value_doc['sensor'] = self.sensor
            value_doc['algorithm_model'] = self.algorithm_model
            value_doc['ts_field'] = self.ts_field
            value_doc['_type'] = "timeseries_anomaly"
            random_string = str(datetime.datetime.now()) + str(random.random())
            random_id = hashlib.md5(random_string.encode()).hexdigest()
            value_doc['_id'] = random_id
            value_doc['unix_timestamp'] = start_timestamp
            value_doc['@timestamp'] = datetime.datetime.utcfromtimestamp(
                start_timestamp).strftime('%Y-%m-%dT%H:%M:%SZ')
            date_time = datetime.datetime.strptime(
                value_doc['@timestamp'], "%Y-%m-%dT%H:%M:%SZ")
            date_index = data_utils.tsa_index + \
                str(date_time.year) + '.' + \
                str(date_time.month) + '.' + str(date_time.day)
            value_doc['_index'] = date_index

            oracle_seq_doc.append(value_doc)
            start_timestamp += 300

        return oracle_seq_doc

    def predict_past(self, X_feature, past_unix_timestamp):
        if int(time.time()) - past_unix_timestamp > 24*86400:
            logging.warning(
                "Must not forecast over 24h past, must be less than 24h")
            return None
        else:
            hours_predict = int(
                (int(time.time()) - past_unix_timestamp) / 3600)

        memory_steps_in = int(self.model.input_shape[1])
        predict_steps_out = int(self.model.layers[1].output_shape[1])
        X_feature_past = X_feature.reshape(1, memory_steps_in, 1)
        if int(X_feature_past.shape[1]) < memory_steps_in:
            logging.error("Necessary length of input is not enough!")
            return None

        oracle_sequence = list()
        if int(hours_predict) > 24:
            hours_predict = 24

        for i in range(int(hours_predict)):
            y_hat = self.model.predict(X_feature_past)
            oracle_sequence += y_hat.reshape(y_hat.shape[1]).tolist()
            X_feature_past = X_feature_past.reshape(
                X_feature_past.shape[1]).tolist() + oracle_sequence[-predict_steps_out:]
            X_feature_past = array(
                X_feature_past[-memory_steps_in:]).reshape(1, memory_steps_in, 1)

        data_utils = DataUtils()
        ts_field_scale = data_utils.params['scaler']['ts_field']
        sensor_scale = data_utils.params['scaler']['sensor']
        oracle_seq_doc = list()
        start_timestamp = past_unix_timestamp - (past_unix_timestamp % 300)
        for value in oracle_sequence:
            value_doc = {}
            value_doc['value'] = value * \
                ts_field_scale[self.ts_field] * sensor_scale[self.sensor]
            value_doc['sensor'] = self.sensor
            value_doc['algorithm_model'] = self.algorithm_model
            value_doc['_type'] = "timeseries_past"
            random_string = str(datetime.datetime.now()) + str(random.random())
            random_id = hashlib.md5(random_string.encode()).hexdigest()
            value_doc['_id'] = random_id
            value_doc['unix_timestamp'] = start_timestamp
            value_doc['@timestamp'] = datetime.datetime.utcfromtimestamp(
                start_timestamp).strftime('%Y-%m-%dT%H:%M:%SZ')
            date_time = datetime.datetime.strptime(
                value_doc['@timestamp'], "%Y-%m-%dT%H:%M:%SZ")
            date_index = data_utils.tsa_index + \
                str(date_time.year) + '.' + \
                str(date_time.month) + '.' + str(date_time.day)
            value_doc['_index'] = date_index

            oracle_seq_doc.append(value_doc)
            start_timestamp += 300

        return oracle_seq_doc


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


def create_stacked_lstm_model(memory_steps_in=144, predict_steps_out=12):
    '''
    WARNING: DO NOT EDIT ANY CODE!
    '''
    model = Sequential()
    model.add(LSTM(1024, activation='tanh', input_shape=(memory_steps_in, 1)))
    model.add(RepeatVector(predict_steps_out))
    model.add(LSTM(1024, activation='tanh', return_sequences=True))
    model.add(TimeDistributed(Dense(1)))
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    return model
