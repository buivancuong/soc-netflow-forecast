import types
import pandas as pd
import logging
from elasticsearch import Elasticsearch
import datetime
import time

from soc_tsa.utils.io_utils.elasticsearch_utils import create_es_index
from soc_tsa.data_input.elasticsearch_query import *
from soc_tsa.utils.io_utils.data_utils import DataUtils


class DataGathering:

    def __init__(self, strategy=None):
        if strategy is not None:
            self.reading_input = types.MethodType(strategy, self)

    def reading_input(self, sensor, algorithm_model, ts_field):
        print("Default method")


def file_input(self, file_path):
    pass


def elasticsearch_input(self, sensor, algorithm_model, ts_field):
    data_utils = DataUtils()
    ts_field_scale = data_utils.params['scaler']['ts_field']
    sensor_scale = data_utils.params['scaler']['sensor']
    es_server = data_utils.elasticsearch['netflow']
    elasticsearch = Elasticsearch([es_server])
    if algorithm_model.split('-')[0] == 'sax':
        using_pandas = True
    else:
        using_pandas = False

    index_list = list()
    for i in range(8):
        date = datetime.datetime.today() - datetime.timedelta(days=i)
        index = create_es_index(sensor=sensor, unix_timestamp=int(
            time.mktime(date.timetuple())))
        index_list.append(index)

    for i in range(len(index_list)):
        try:
            index_list[i] = check_es_index(
                es_server=es_server, index=index_list[i])
        except:
            logging.error("Cannot connect to Elasticsearch")
        if index_list[i] == '':
            logging.warning("No have index data at index " + index_list[i])
    if ''.join(index_list):
        index_search = ''
        for i in index_list:
            if i:
                index_search += (',' + i)
        index_search = index_search[1:]
    else:
        logging.error("No have input data for 7 days ago")
        return None

    es_query_body = create_es_query(
        sensor=sensor, ts_field=ts_field, purpose='train')
    result_training_sequence = list()
    result_training_dataframe = pd.DataFrame(columns=['timestamp', 'value'])

    try:
        es_json_result = elasticsearch.search(
            index=index_search, body=es_query_body, request_timeout=30)
    except Exception as err:
        logging.error(
            "Cannot connect to Elasticsearch to get training timeseries data")
        logging.error(err)
        return None
    for record in es_json_result['aggregations']['2']['buckets']:
        result_training_sequence.append(
            record['1']['value'] / (ts_field_scale[ts_field] * sensor_scale[sensor]))
        if using_pandas:
            result_training_dataframe = result_training_dataframe.append({
                "timestamp": record['key'] / 1000,
                "value": record['1']['value']},
                ignore_index=True)
    if using_pandas:
        # result_training_dataframe.to_csv("instance.csv")
        return result_training_dataframe
    return np.array(result_training_sequence)
