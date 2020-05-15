from statsmodels.tsa.arima_model import ARIMA, ARMAResults
import pickle


class Arima():
    def __init__(self):
      return None

    def fit(self, trend, p, d, q):
        best_score = float("inf")
        for _p in range(2, p):
            for _q in range(2, q):
                for _d in range(0, d+1):
                    order = (_p, _d, _q)
                    try:
                        model = ARIMA(trend, order)
                        model_arima = model.fit()
                        score = model_arima.aic
                        if score < best_score:
                            best_score, best_cfg = score, order
                    except:
                        self.fit(trend, _p, _d, _q-1)
                        with open('arima.pkl', 'wb') as pkl:
                            pickle.dump(model_arima, pkl)
                    else:
                        with open('arima.pkl', 'wb') as pkl:
                            pickle.dump(model_arima, pkl)
