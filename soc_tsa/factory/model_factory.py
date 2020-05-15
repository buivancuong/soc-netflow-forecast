from soc_tsa.utils.io_utils.data_utils import DataUtils
from soc_tsa.models.autoencoder.stacked_lstm.stacked_lstm import AE_LSTM
from soc_tsa.models.statistical.sax_lstm.sax_lstm import SAX_LSTM

import logging


class ModelFactory:

    # @staticmethod
    def __init__(self):
        pass

    def generate_model(self, sensor, algorithm_model, ts_field):
        algorithm, model = algorithm_model.split('-')

        if algorithm == 'ae':
            if model == 'lstm':
                return AE_LSTM(sensor=sensor, algorithm_model=algorithm_model, ts_field=ts_field)
            else:
                logging.error(
                    "Model of Autoencoder has not been supported yet")
                return None
        elif algorithm == 'sax':
            if model == 'lstm':
                return SAX_LSTM(sensor=sensor, algorithm_model=algorithm_model, ts_field=ts_field)
            else:
                logging.error("Model of Statistic has not been supported yet")
                return None
        else:
            logging.error("Model has not been supported yet")
