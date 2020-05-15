import schedule
import time
import datetime
import requests
import logging
import json
import hashlib
import random
from elasticsearch import Elasticsearch

from soc_tsa.utils.io_utils.data_utils import DataUtils
from soc_tsa.data_input.elasticsearch_query import get_current_time_point


def get_current_ts_bucket(sensor, ts_field):
    current_point = get_current_time_point(sensor=sensor, ts_field=ts_field)
    if current_point is None:
        logging.error("Unable to update last value of current time point")

    else:
        real_doc = list()
        data_utils = DataUtils()
        timestamp = int(time.time()) - (int(time.time() % 300)) - 300
        value_doc = {}
        value_doc['value'] = current_point['1']['value']
        value_doc['sensor'] = sensor
        value_doc['algorithm_model'] = "real_data"
        value_doc['ts_field'] = ts_field
        value_doc['_type'] = "timeseries_anomaly"
        random_string = str(datetime.datetime.now()) + str(random.random())
        random_id = hashlib.md5(random_string.encode()).hexdigest()
        value_doc['_id'] = random_id
        value_doc['unix_timestamp'] = timestamp
        value_doc['@timestamp'] = datetime.datetime.utcfromtimestamp(
            timestamp).strftime('%Y-%m-%dT%H:%M:%SZ')
        date_time = datetime.datetime.strptime(
            value_doc['@timestamp'], "%Y-%m-%dT%H:%M:%SZ")
        date_index = data_utils.tsa_index + \
            str(date_time.year) + '.' + \
            str(date_time.month) + '.' + str(date_time.day)
        value_doc['_index'] = date_index
        real_doc.append(value_doc)
        print(value_doc)
        data_utils.post_timeseries_bulk(real_doc)


with open('./config.json') as config:
    config = json.load(config)
    sensor_list = config['params']['sensor']
    ts_field_list = config['params']['ts_field']

while 1:
    for sensor in sensor_list:
        for ts_field in ts_field_list:
            get_current_ts_bucket(sensor=sensor, ts_field=ts_field)
    print(300 - int(time.time() % 300))
    time.sleep(300 - int(time.time() % 300))
    
