from soc_tsa.utils.io_utils.data_utils import DataUtils
from soc_tsa.data_input.elasticsearch_query import *
from soc_tsa.utils.io_utils.sqlite_access import SQLiteDB
from soc_tsa.factory.model_factory import ModelFactory
from soc_tsa.factory.data_output_factory import DataHandlerFactory
import pandas as pd

from keras.backend import clear_session
from keras.backend.tensorflow_backend import clear_session
from keras.backend.tensorflow_backend import get_session
import datetime
import json
from flask import Flask
from flask import Response
from flask import request
from flask import jsonify
import logging

logging.basicConfig(filename='app.log', filemode='w', level=logging.INFO,
                    format='%(name)s - %(levelname)s - %(message)s')


app = Flask(__name__)

netflow_field = {
    "in-bytes": "IN_BYTES",
    "out-bytes": "OUT_BYTES",
    "in-pkts": "IN_PKTS",
    "out-pkts": "OUT_PKTS"
}


@app.route('/api/v1/models', methods=['GET'])
def get_models_list():
    '''
    GET /api/v1/models
    '''

    data_utils = DataUtils()

    sqlite_db = SQLiteDB(data_utils.sqlite_dir)
    sqlite_db.create_db_connection()
    db_result = sqlite_db.list_models("models")
    json_result = {}
    for record in db_result:
        json_result.update(record[0])
    return jsonify(json_result)


@app.route('/api/v1/models', methods=['POST'])
def create_model():
    '''
    POST /api/v1/models
    {
        ... json object: info of model that wish create
    }
    sensor: sensor-xxx
    algorithm_model:    ae-lstm (autoencoder + lstm)
                        vae-lstm (variational autoencoder + lstm)
                        sax-lstm (statistical)
    ts_field: IN_BYTES, OUT_BYTES, IN_PKTS, OUT_PKTS
    '''
    clear_session()
    body = request.get_json()
    print(json.dumps(body))

    try:
        sensor = body['sensor']
    except:
        logging.error("Misssing value of sensor name: sensor")
        return jsonify({"ERROR": "Missing parameter"})

    try:
        algorithm_model = body['algorithm_model']
    except:
        logging.error("Misssing value of algorithm: algorithm_model")
        return jsonify({"ERROR": "Missing parameter"})

    try:
        ts_field = body['ts_field']
    except:
        logging.error("Misssing value of timeseries field: ts_field")
        return jsonify({"ERROR": "Missing parameter"})

    if not DataUtils.check_input(sensor, algorithm_model, ts_field):
        logging.error("Invalid input data to create new model")
        resp = Response(json.dumps(
            {"Message": "Invalid input data to create new model"}))
        resp.status_code = 400
        return resp

    model_factory = ModelFactory()
    model_object = model_factory.generate_model(
        sensor=sensor, algorithm_model=algorithm_model, ts_field=ts_field)

    json_result = {
        "model_id": model_object.model_id,
        "model_path": model_object.model_path,
    }
    clear_session()
    return jsonify(json_result)


@app.route('/api/v1/models/', methods=['PUT'])
def training_model():
    '''
    PUT api/v1/models/
    {
        "model_id": "sensor-xxx_ae-lstm_in-bytes",
    }
    '''

    clear_session()
    body = request.get_json()
    print(json.dumps(body))
    try:
        model_id = body['model_id']
    except:
        logging.error("No have value of model id: model_id")
        return jsonify({"Error": "Missing parameter"})

    if len(model_id.split('_')) != 3:
        logging.error("Fail to extract model ID to training model")
        resp = Response(json.dumps(
            {"Message": "Invalid model ID to training model"}))
        resp.status_code = 400
        return resp

    sensor, algorithm_model, ts_field = model_id.split('_')

    if not DataUtils.check_input(sensor, algorithm_model, ts_field):
        logging.error("Invalid input model")
        resp = Response(json.dumps({"Message": "Invalid input model"}))
        resp.status_code = 400
        return resp

    data_output_factory = DataHandlerFactory()
    data_output = data_output_factory.generate_data_handler(
        sensor=sensor, algorithm_model=algorithm_model, ts_field=ts_field)
    
    training_data = data_output.handle_data(
        sensor=sensor, algorithm_model=algorithm_model, ts_field=ts_field)

    if training_data is None:
        logging.error("The training data is Node")
        resp = Response(json.dumps({"Message": "Invalid input data"}))
        resp.status_code = 400
        return resp

    model_factory = ModelFactory()
    model_object = model_factory.generate_model(
        sensor=sensor, algorithm_model=algorithm_model, ts_field=ts_field)
    model_object.training_model(training_data=training_data)

    response = {
        "model_id": model_id,
        # "model_path": model_object.model_path,
        "mse_loss": "loss of model after training"
    }
    clear_session()
    return response


# @app.route('/api/v1/models/', methods=['PUT'])
# def training_model():
#     '''
#     PUT api/v1/models/
#     {
#         "model_id": "sensor-xxx_ae-lstm_in-bytes",
#     }
#     '''
#     clear_session()
#     body = request.get_json()
#     print(json.dumps(body))
#     try:
#         model_id = body['model_id']
#     except:
#         logging.error("No have value of model id: model_id")
#         return jsonify({"Error": "Missing parameter"})

#     if len(model_id.split('_')) != 3:
#         logging.error("Fail to extract model ID to training model")
#         resp = Response(json.dumps(
#             {"Message": "Invalid model ID to training model"}))
#         resp.status_code = 400
#         return resp

#     sensor, algorithm_model, ts_field = model_id.split('_')

#     if not DataUtils.check_input(sensor, algorithm_model, ts_field):
#         logging.error("Invalid input model")
#         resp = Response(json.dumps({"Message": "Invalid input model"}))
#         resp.status_code = 400
#         return resp

#     model_factory = ModelFactory()
#     model_object = model_factory.generate_model(
#         sensor=sensor, algorithm_model=algorithm_model, ts_field=ts_field)

#     data_gathering = DataGathering(strategy=elasticsearch_input)
#     training_data = data_gathering.reading_input(
#         sensor=sensor, algorithm_model=algorithm_model, ts_field=ts_field)
#     if training_data is None:
#         clear_session()
#         logging.error("No exist data to training model")
#         return jsonify({"ERROR": "No exist data to training model"})
#     memory_steps_in = model_object.model.layers[0].input_shape[1]
#     predict_steps_out = model_object.model.layers[3].output_shape[1]
#     X_feature, y_label = split_sequence_stacked_lstm(
#         data_sequence=training_data.tolist(), memory_steps_in=memory_steps_in, predict_steps_out=predict_steps_out)
#     X_feature = X_feature.reshape(X_feature.shape[0], X_feature.shape[1], 1)
#     y_label = y_label.reshape(y_label.shape[0], y_label.shape[1], 1)

#     model_object.training_model(training_data=training_data)

#     response = {
#         "model_id": model_id,
#         "model_path": model_object.model_path,
#         "mse_loss": "loss of model after training"
#     }
#     clear_session()
#     return response


@app.route('/api/v1/models/forecast/')
def forcast():
    '''
    GET api/v1/models/forecast/?model_id=blablabla&forecast_hours=1
    '''
    clear_session()
    model_id = request.args.get('model_id')
    forecast_hours = int(request.args.get('forecast_hours'))
    if forecast_hours > 24:
        logging.warning("Bo deo cho doan qua 24h")
        return "Bad Request", 400

    if len(model_id.split('_')) != 3:
        logging.error("Fail to extract model ID to forecast")
        resp = Response(json.dumps(
            {"Message": "Invalid model ID to forecast"}))
        resp.status_code = 400
        return resp

    sensor, algorithm_model, ts_field = model_id.split('_')

    if not DataUtils.check_input(sensor, algorithm_model, ts_field):
        logging.error("Invalid input model")
        resp = Response(json.dumps({"Message": "Invalid input model"}))
        resp.status_code = 400
        return resp

    model_factory = ModelFactory()
    model_object = model_factory.generate_model(
        sensor=sensor, algorithm_model=algorithm_model, ts_field=ts_field)

    current_timeseries = get_current_timeseries(
        sensor=sensor, ts_field=ts_field)
    if current_timeseries is None:
        clear_session()
        logging.error("No current data to predict")
        return jsonify({"ERROR": "No current data to predict"})
    oracle_seq_list = model_object.prediction(
        data_input=current_timeseries, hours_predict=forecast_hours)
    clear_session()

    data_utils = DataUtils()
    data_utils.post_timeseries_bulk(ts_bulk=oracle_seq_list)
    data_utils.post_timeseries_bulk(
        ts_bulk=data_utils.create_lowwer_theshole(ts_bulk=oracle_seq_list))
    data_utils.post_timeseries_bulk(
        ts_bulk=data_utils.create_upper_threshole(ts_bulk=oracle_seq_list))
    return str(oracle_seq_list)


@app.route("/api/v1/models/forecast_past/")
def forecast_past():
    '''
    GET api/v1/models/forecast/?model_id=blablabla&past_unixtime=$UNIXTIMESTAMP
    '''
    clear_session()
    model_id = request.args.get('model_id')
    if len(model_id.split('_')) != 3:
        logging.error("Fail to extract model ID to training model")
        resp = Response(json.dumps(
            {"Message": "Invalid model ID to training model"}))
        resp.status_code = 400
        return resp

    sensor, algorithm_model, ts_field = model_id.split('_')

    if not DataUtils.check_input(sensor, algorithm_model, ts_field):
        logging.error("Invalid input data")
        resp = Response(json.dumps({"Message": "Invalid input data"}))
        resp.status_code = 400
        return resp
    past_unix_timestamp = int(request.args.get('from'))

    model_factory = ModelFactory()
    model_object = model_factory.generate_model(
        sensor=sensor, algorithm_model=algorithm_model, ts_field=ts_field)

    current_past_timeseries = get_current_past_timeseries(
        sensor=sensor, ts_field=ts_field, past_unix_timestamp=past_unix_timestamp)

    if current_past_timeseries is None:
        clear_session()
        logging.error("No past data to predict")
        return jsonify({"ERROR": "No pass data to predict"})
    oracle_seq_list = model_object.predict_past(
        X_feature=current_past_timeseries, past_unix_timestamp=past_unix_timestamp)
    clear_session()

    data_utils = DataUtils()
    data_utils.post_timeseries_bulk(ts_bulk=oracle_seq_list)
    return str(oracle_seq_list)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
