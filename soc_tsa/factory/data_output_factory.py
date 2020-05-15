from soc_tsa.utils.io_utils.data_utils import DataUtils
from soc_tsa.data_output.dlearn_model import DLearnModel, algorithm_is_ae
from soc_tsa.data_output.stat_model import StatModel, algorithm_is_sax

import logging

class DataHandlerFactory:

    # @staticmethod
    def __init__(self):
        pass

    def generate_data_handler(self, sensor, algorithm_model, ts_field):
        algorithm = algorithm_model.split('-')[0]

        if algorithm == 'ae':
            return DLearnModel(strategy=algorithm_is_ae)
        elif algorithm == 'sax':
            return StatModel(strategy=algorithm_is_sax)
        else:
            logging.error("Model has not been supported yet")
            return None