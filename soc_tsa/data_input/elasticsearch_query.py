from elasticsearch import Elasticsearch
import pandas as pd
import numpy as np
import json
import datetime
import requests
import logging
import time

from soc_tsa.utils.io_utils.data_utils import DataUtils
from soc_tsa.utils.io_utils.elasticsearch_utils import create_es_index


def create_es_query(sensor, ts_field, purpose):
    data_utils = DataUtils()
    timeseries_field = data_utils.elasticsearch['ts_field']
    interval_minutes = str(int(data_utils.interval / 60)) + 'm'
    if purpose == "train":
        query_body = {
            "aggs": {
                "2": {
                    "date_histogram": {
                        "field": "@timestamp",
                        "interval": interval_minutes
                    },
                    "aggs": {
                        "1": {
                            "sum": {
                                "field": timeseries_field[ts_field]
                            }
                        }
                    }
                }
            },
            "size": 0,
            "query": {
                "bool": {
                    "must": [
                        {
                            "match": {
                                "NTOPNG_INSTANCE_NAME.keyword": sensor
                            }
                        }
                    ]
                }
            }
        }
    elif purpose == "forecast":
        query_body = {
            "aggs": {
                "2": {
                    "date_histogram": {
                        "field": "@timestamp",
                        "interval": interval_minutes
                    },
                    "aggs": {
                        "1": {
                            "sum": {
                                "field": timeseries_field[ts_field]
                            }
                        }
                    }
                }
            },
            "size": 0,
            "query": {
                "bool": {
                    "must": [
                        {
                            "match": {
                                "NTOPNG_INSTANCE_NAME.keyword": sensor
                            }
                        },
                        {
                            "range": {
                                "@timestamp": {
                                    "gte": "now-13h"
                                }
                            }
                        }
                    ]
                }
            }
        }
    elif purpose == "update":
        query_body = {
            "aggs": {
                "2": {
                    "date_histogram": {
                        "field": "@timestamp",
                        "interval": interval_minutes
                    },
                    "aggs": {
                        "1": {
                            "sum": {
                                "field": timeseries_field[ts_field]
                            }
                        }
                    }
                }
            },
            "size": 0,
            "query": {
                "bool": {
                    "must": [
                        {
                            "match": {
                                "NTOPNG_INSTANCE_NAME.keyword": sensor
                            }
                        },
                        {
                            "range": {
                                "@timestamp": {
                                    "gte": "now-1h"
                                }
                            }
                        }
                    ]
                }
            }
        }
    else:
        logging.warning("Error in purpose")
        return None
    return json.dumps(query_body)


def check_es_index(es_server, index):
    '''
    Perform a check to verify the existing of elasticsearch index before search
    '''
    response = requests.head(
        url=es_server + index
    )
    if response.status_code == 200:
        return index
    logging.warning("That day no have index")
    return ''


def get_current_timeseries(sensor, ts_field):
    data_utils = DataUtils()
    es_server = data_utils.elasticsearch['netflow']
    ts_field_scale = data_utils.params['scaler']['ts_field']
    sensor_scale = data_utils.params['scaler']['sensor']
    elasticsearch = Elasticsearch([es_server])
    today = create_es_index(
        sensor=sensor, unix_timestamp=time.mktime(datetime.datetime.today().timetuple()))
    yesterday = create_es_index(
        sensor=sensor, unix_timestamp=time.mktime((datetime.datetime.today() - datetime.timedelta(days=1)).timetuple()))
    es_query_body = create_es_query(
        sensor=sensor, ts_field=ts_field, purpose='forecast')
    try:
        today = check_es_index(
            es_server=es_server, index=today)
        yesterday = check_es_index(
            es_server=es_server, index=yesterday)
        if today == yesterday and yesterday == '':
            logging.warning('No have index data on last 24h')
            return None
    except:
        logging.error("Cannot connect to Elasticsearch")
        return None
    try:
        es_json_result = elasticsearch.search(
            index=yesterday+','+today, body=es_query_body, request_timeout=20)
    except Exception as error:
        logging.error(
            "Cannot connect to Elasticsearch to get current timeseries data")
        logging.error(error)
        return None
    result_current_sequence = list()
    if len(es_json_result['aggregations']['2']['buckets']) > 144:
        for record in es_json_result['aggregations']['2']['buckets'][-145:-1]:
            result_current_sequence.append(
                record['1']['value'] / (ts_field_scale[ts_field] * sensor_scale[sensor]))
    else:
        logging.error("Length of past input data is not exact")
        return None
    return np.array(result_current_sequence)


def get_current_past_timeseries(sensor, ts_field, past_unix_timestamp):
    if int(time.time()) - past_unix_timestamp > 24*86400:
        logging.warning(
            "Must not forecast over 24h past, must be less than 24h")
        return None

    data_utils = DataUtils()
    es_server = data_utils.elasticsearch['netflow']
    timeseries_field = data_utils.elasticsearch['ts_field']
    ts_field_scale = data_utils.params['scaler']['ts_field']
    sensor_scale = data_utils.params['scaler']['sensor']
    elasticsearch = Elasticsearch([es_server])
    past_hours_index = create_es_index(
        sensor=sensor, unix_timestamp=past_unix_timestamp)
    prev_day_index = create_es_index(sensor=sensor, unix_timestamp=(
        past_unix_timestamp - 86400))
    try:
        past = check_es_index(es_server=es_server, index=past_hours_index)
        prev = check_es_index(
            es_server=es_server, index=prev_day_index)
        if past == prev and prev == '':
            logging.warning('No have index data on last 24h pass')
            return None
    except Exception as err:
        logging.error("Cannot connect to Elasticsearch")
        logging.error(err)
        return None
    query_body = {
        "aggs": {
            "2": {
                "date_histogram": {
                    "field": "@timestamp",
                    "interval": "5m"
                },
                "aggs": {
                    "1": {
                        "sum": {
                            "field": timeseries_field[ts_field]
                        }
                    }
                }
            }
        },
        "size": 0,
        "query": {
            "bool": {
                "must": [
                    {
                        "match": {
                            "NTOPNG_INSTANCE_NAME.keyword": sensor
                        }
                    },
                    {
                        "range": {
                            "@timestamp": {
                                "gte": past_unix_timestamp - (past_unix_timestamp % 300) - 43200,
                                "lte": past_unix_timestamp - (past_unix_timestamp % 300),
                                "format": "epoch_second"
                            }
                        }
                    }
                ]
            }
        }
    }
    json_response = elasticsearch.search(
        index=prev+','+past, body=query_body, request_timeout=20)

    result_past_sequence = list()
    if len(json_response['aggregations']['2']['buckets']) == 144:
        for record in json_response['aggregations']['2']['buckets']:
            result_past_sequence.append(
                record['1']['value'] / (ts_field_scale[ts_field] * sensor_scale[sensor]))
    else:
        logging.error("Length of past input data is not exact")
        return None
    return np.array(result_past_sequence)

def get_current_time_point(sensor, ts_field):
    data_utils = DataUtils()
    es_server = data_utils.elasticsearch['netflow']
    elasticsearch = Elasticsearch([es_server])
    today = create_es_index(
        sensor=sensor, unix_timestamp=time.mktime(datetime.datetime.today().timetuple()))
    yesterday = create_es_index(
        sensor=sensor, unix_timestamp=time.mktime((datetime.datetime.today() - datetime.timedelta(days=1)).timetuple()))
    es_query_body = create_es_query(
        sensor=sensor, ts_field=ts_field, purpose='update')
    try:
        today = check_es_index(
            es_server=es_server, index=today)
        yesterday = check_es_index(
            es_server=es_server, index=yesterday)
        if today == yesterday and yesterday == '':
            logging.warning('No have index data on last 24h')
            return None
    except:
        logging.error("Cannot connect to Elasticsearch")
        return None
    try:
        es_json_result = elasticsearch.search(
            index=yesterday+','+today, body=es_query_body, request_timeout=20)
    except Exception as error:
        logging.error(
            "Cannot connect to Elasticsearch to get current timeseries data")
        logging.error(error)
        return None
    
    if len(es_json_result['aggregations']['2']['buckets']) > 1:
        return es_json_result['aggregations']['2']['buckets'][-2]
    else:
        logging.error("Have no data of last point interval")
        return None