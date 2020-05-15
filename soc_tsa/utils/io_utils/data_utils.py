import json
import os
import datetime
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import requests
import statistics


class DataUtils:

    def __init__(self):
        with open('./config.json') as config_file:
            config = json.load(config_file)
            self.data_dir = config['dir']['data_dir']
            self.models_dir = config['dir']['models_dir']
            self.elasticsearch = config['elasticsearch']
            self.sqlite_dir = config['dir']['sqlite']
            self.interval = config['model']['bucket_interval']
            self.tsa_es = config['elasticsearch']['tsa_es']
            self.tsa_index = config['elasticsearch']['tsa_index']
            self.params = config['params']

    @staticmethod
    def post_timeseries_bulk(ts_bulk):
        data_utils = DataUtils()
        today = datetime.datetime.now()
        suffix = str(today.year) + '.' + \
            str(today.month) + '.' + str(today.day)
        es = Elasticsearch([data_utils.tsa_es])
        if check_elasticsearc_index_status(es_server=data_utils.tsa_es, index=data_utils.tsa_index+suffix) == '':
            url_new_index = data_utils.tsa_es + data_utils.tsa_index + suffix
            requests.put(url=url_new_index)
        bulk(client=es, actions=ts_bulk)

    @staticmethod
    def check_input(sensor, algorithm_model, ts_field):
        data_utils = DataUtils()
        if data_utils.params['sensor'].count(sensor) == 0:
            return False
        if data_utils.params['algorithm_model'].count(algorithm_model) == 0:
            return False
        if data_utils.params['ts_field'].count(ts_field) == 0:
            return False
        return True
    
    @staticmethod
    def create_upper_threshole(ts_bulk):
        values = [value['value'] for value in ts_bulk]
        stddev = statistics.stdev(values)
        for value in ts_bulk:
            value['value'] += stddev
            value['_type'] = 'upper_threshole'
        return ts_bulk
    
    @staticmethod
    def create_lowwer_theshole(ts_bulk):
        values = [value['value'] for value in ts_bulk]
        stddev = statistics.stdev(values)
        for value in ts_bulk:
            value['value'] -= stddev
            value['_type'] = 'lowwer_threshole'
        return ts_bulk


    
def check_elasticsearc_index_status(es_server, index):
    '''
    Perform a check to verify the existing of elasticsearch index before search
    '''
    response = requests.head(
        url=es_server + index
    )
    if response.status_code == 200:
        return index
    return ''
