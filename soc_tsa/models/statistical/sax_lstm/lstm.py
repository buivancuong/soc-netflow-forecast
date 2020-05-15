from keras.layers import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
from keras import optimizers
import pickle

class lstm():
    def __init__(self):
        return None

    def create_lstm(self, n_features, neurons):
        pre_model = Sequential()
        pre_model.add(LSTM(neurons, input_shape=(1, n_features)))
        # pre_model.add(Dropout(p=0.2))
        # pre_model.add(Dense(neurons))
        # pre_model.add(Dropout(p=0.2))
        pre_model.add(Dense(1))
        adam = optimizers.Adam(
            lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        pre_model.compile(loss='mean_squared_error', optimizer=adam)
        return pre_model

    def fit(self, X_train, y_train, neurons, nb_epoch, batch_size):
        n_features = X_train.shape[2]
        pre_model = self.create_lstm(n_features, neurons)
        for i in range(nb_epoch):
            pre_model.fit(X_train, y_train, epochs=1,
                          batch_size=batch_size, verbose=0, shuffle=False)
            pre_model.reset_states()

        with open('lstm_model.pkl', 'wb') as handle:
            pickle.dump(pre_model, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def predict(self, X_test, n_step):
        yhat_lstm = []
        for i in range(n_step):
            temp = X_test.copy()
            with open('lstm_model.pkl', 'rb') as pkl:
                yhat = pickle.load(pkl).predict(
                    X_test.reshape(1, 1, X_test.shape[1]))
            yhat_lstm.append(float(yhat))
            # Add yhat -> X_test
            for k in range(1, temp.shape[1]):
                X_test[0][k] = temp[0][k-1].copy()
            X_test[0][0] = yhat
        return yhat_lstm

def create_lstm(self, n_features):
        pre_model = Sequential()
        pre_model.add(LSTM(1, input_shape=(1, n_features)))
        # pre_model.add(Dropout(p=0.2))
        # pre_model.add(Dense(1))
        # pre_model.add(Dropout(p=0.2))
        pre_model.add(Dense(12))
        adam = optimizers.Adam(
            lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        pre_model.compile(loss='mean_squared_error', optimizer=adam)
        return pre_model