from soc_tsa.data_output.data_output import DataOutput
import types
from soc_tsa.utils.io_utils.data_utils import DataUtils
from soc_tsa.data_input.data_input import DataGathering, elasticsearch_input

import logging


class StatModel(DataOutput):

    def __init__(self, strategy=None):
        if strategy is not None:
            self.handle_data = types.MethodType(strategy, self)

    def handle_data(self, sensor, algorithm_model, ts_field):
        print("Default method DLearnModel")
        pass


def algorithm_is_sax(self, sensor, algorithm_model, ts_field):
    data_input = DataGathering(strategy=elasticsearch_input)

    original_timeseries_numpy_array = data_input.reading_input(
        sensor=sensor, algorithm_model=algorithm_model, ts_field=ts_field)

    if original_timeseries_numpy_array is None:
        logging.error('No exist data to training model')
        return None

    return original_timeseries_numpy_array
