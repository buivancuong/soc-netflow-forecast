import schedule
import time
import datetime
import requests
import logging
import json

with open('./config.json') as config:
    config = json.load(config)
    sensor_list = config['params']['sensor']
    algorithm_model_list = config['params']['algorithm_model'][1:]
    ts_field_list = config['params']['ts_field']

# sensor_list = ['sensor-abbank-01']
# algorithm_model_list = ['ae-lstm']
# ts_field_list = ['in-bytes']


SOC_TSA_TRAIN = 'http://localhost:5000/api/v1/models/'
SOC_TSA_FORECAST = 'http://localhost:5000/api/v1/models/forecast/'


def training():
    for sensor in sensor_list:
        for algorithm_model in algorithm_model_list:
            for ts_field in ts_field_list:
                model_id = sensor + '_' + algorithm_model + '_' + ts_field
                response = requests.put(
                    url=SOC_TSA_TRAIN,
                    data=json.dumps({
                        "model_id": model_id
                    }),
                    headers={"Content-Type": "application/json"}
                )
                if response.status_code != 200:
                    logging.error("training error " + model_id +
                                  ": " + str(response.status_code))
                else:
                    logging.warning("training warning " + model_id +
                                    ": " + str(response.status_code))
    logging.warning("Over 1 period all models")


def forecast(forecast_hours=24):
    for sensor in sensor_list:
        for algorithm_model in algorithm_model_list:
            for ts_field in ts_field_list:
                model_id = sensor + '_' + algorithm_model + '_' + ts_field
                full_url = SOC_TSA_FORECAST + '?model_id=' + model_id + \
                    '&forecast_hours=' + str(forecast_hours)
                response = requests.get(url=full_url)
                if response.status_code != 200:
                    logging.error("forecast error " + model_id +
                                  ": " + str(response.status_code))
                else:
                    logging.warning("forecast warning " + model_id +
                                    ": " + str(response.status_code))
    logging.warning("Over 1 period all models")


while 1:
    # forecast()
    training()
    time.sleep(12*3600)
