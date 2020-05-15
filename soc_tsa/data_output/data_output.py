from soc_tsa.utils.io_utils.data_utils import DataUtils
import logging
import datetime
import time
import pandas as pd


class DataOutput:

    def handle_data(self, sensor, algorithm_model, ts_field):
        pass


def fill_0(dataframe, purpose):
    filled_dataframe = pd.DataFrame(columns=['timestamp', 'value'])
    if purpose == 'train':
        start_timestamp = dataframe['timestamp'][0]
        today = datetime.datetime.today()
        end_timestamp = int(time.mktime(datetime.datetime(
            today.year, today.month, today.day).timetuple()))
        while start_timestamp < end_timestamp:
            if start_timestamp == dataframe['timestamp'][0]:
                filled_dataframe = filled_dataframe.append(
                    {"timestamp": dataframe['timestamp'][0], "value": dataframe['value'][0]}, ignore_index=True)
            else:
                filled_dataframe = filled_dataframe.append(
                    {"timestamp": start_timestamp, "value": 0}, ignore_index=True)
            dataframe.drop(dataframe.index[0])
            start_timestamp += 300
        while len(dataframe):
            filled_dataframe = filled_dataframe.append(
                {"timestamp": dataframe['timestamp'][0], "value": dataframe['value'][0]}, ignore_index=True)
            dataframe.drop(dataframe.index[0])
        return filled_dataframe
    elif purpose == 'forecast':
        pass
    else:
        logging.error("Error in purpose parameter")
        return None
