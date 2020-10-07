import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.layers import LSTM

# Please change the file location before running this code.
ann_con = "C:/Users/Prin/Desktop/dissertation/Datashare/ann_condition.csv"
lstm_con = "C:/Users/Prin/Desktop/dissertation/Datashare/lstm_condition.csv"

'''
Single-step NN model
'''
# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence) - 1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def plot_one(test, pred, mod):
    plt.figure(figsize=(30, 6))
    plt.plot(test, label='True')
    plt.plot(pred, label=mod + ' forecasting')
    plt.title(mod + "'s Prediction")
    plt.xlabel('Observation')
    plt.ylabel('Total occupants')
    plt.legend()
    plt.show()

def one_step_nn(hist, period, col, model):
    step_in = 1
    df = hist[hist.Term_dates == period]
    X = df[[col]]

    train_size = int(len(X) * 0.6)
    train, test = X[0:train_size], X[train_size:]

    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_sc = scaler.fit_transform(train)
    test_sc = scaler.transform(test)

    '''
    ANN model
    '''
    if model == 'ann':

        checker = True
        try:
            ann_param = pd.read_csv(ann_con)
        except:
            checker = False
            print('Error: File "ann_condition.csv" cannot be found.')
            print('Please check the file location.')

        if checker == False:
            print('Stop predicting.')
            return 0, 0, 0

        else:
            search_param = ann_param[ann_param['Term Date'] == period]

            init = search_param['Initializer'].values[0]
            epoch = search_param['epoch'].values[0]
            neuron = search_param['nueron'].values[0]

            n_steps_in = step_in

            train_ann_sc_list = [i[0] for i in train_sc.tolist()]
            test_ann_sc_list = [i[0] for i in test_sc.tolist()]

            # split into samples
            X_train, y_train = split_sequence(train_ann_sc_list, n_steps_in)
            X_test, y_test = split_sequence(test_ann_sc_list, n_steps_in)

            nn_model = Sequential()
            nn_model.add(Dense(neuron, input_dim=n_steps_in, activation='relu', kernel_initializer=init))
            nn_model.add(Dense(1))
            nn_model.compile(loss='mean_squared_error', optimizer='adam')
            early_stop = EarlyStopping(monitor='loss', patience=2, verbose=1)
            nn_model.fit(X_train, y_train, epochs=epoch, batch_size=1, verbose=1, callbacks=[early_stop], shuffle=False)

            y_pred_test_nn = nn_model.predict(X_test)
            y_pred_test_nn_inver = scaler.inverse_transform(y_pred_test_nn)
            y_test_inver = scaler.inverse_transform(y_test.reshape(-1, 1))

            MAE = metrics.mean_absolute_error(y_test_inver, y_pred_test_nn_inver)
            MSE = metrics.mean_squared_error(y_test_inver, y_pred_test_nn_inver)
            RMSE = np.sqrt(metrics.mean_squared_error(y_test_inver, y_pred_test_nn_inver))
            print('ANN MAE:', MAE)
            print('ANN MSE:', MSE)
            print('ANN RMSE:', RMSE)

            plot_one(y_test_inver, y_pred_test_nn_inver, 'ANN')

            return MAE, MSE, RMSE

    '''
    LSTM model
    '''
    if model == 'lstm':

        checker = True
        try:
            lstm_param = pd.read_csv(lstm_con)
        except:
            checker = False
            print('Error: File "lstm_condition.csv" cannot be found.')
            print('Please check the file location.')

        if checker == False:
            print('Stop predicting.')
            return 0, 0, 0

        else:
            search_param = lstm_param[lstm_param['Term Date'] == period]

            init = search_param['Initializer'].values[0]
            epoch = search_param['epoch'].values[0]
            neuron = search_param['nueron'].values[0]

            n_steps_in = step_in

            X_train_lmse, y_train = split_sequence(train_sc, n_steps_in)
            X_test_lmse, y_test = split_sequence(test_sc, n_steps_in)

            lstm_model = Sequential()
            lstm_model.add(LSTM(neuron, input_shape=(n_steps_in, 1), activation='relu', kernel_initializer=init,
                                return_sequences=False))
            lstm_model.add(Dense(1))
            lstm_model.compile(loss='mean_squared_error', optimizer='adam')
            early_stop = EarlyStopping(monitor='loss', patience=2, verbose=1)
            lstm_model.fit(X_train_lmse, y_train, epochs=epoch, batch_size=1, verbose=1, shuffle=False,
                        callbacks=[early_stop])

            y_pred_test_lstm = lstm_model.predict(X_test_lmse)
            y_pred_test_lstm_inver = scaler.inverse_transform(y_pred_test_lstm)
            y_test_inver = scaler.inverse_transform(y_test)

            MAE = metrics.mean_absolute_error(y_test_inver, y_pred_test_lstm_inver)
            MSE = metrics.mean_squared_error(y_test_inver, y_pred_test_lstm_inver)
            RMSE = np.sqrt(metrics.mean_squared_error(y_test_inver, y_pred_test_lstm_inver))

            print('LSTM MAE:', MAE)
            print('LSTM MSE:', MSE)
            print('LSTM RMSE:', RMSE)
            plot_one(y_test_inver, y_pred_test_lstm_inver, 'LSTM')

            return MAE, MSE, RMSE


'''
Multi-step NN model
'''

def split_multi_step(sequence, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # check if we are beyond the sequence
        if out_end_ix > len(sequence):
            print("Too much number of step_in")
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


def plot_result(pred_df, act_df):
    ax = act_df.plot(label='History')
    pred_df.plot(ax=ax, label='Forecast', alpha=.7, figsize=(20, 7))
    ax.set_xlabel('Date')
    ax.set_ylabel('Total amount of user')
    plt.legend()
    plt.show()


def multi_step_nn(train, test, step_in, step_out, model, epoch, neuron, init):
    if step_in < len(train):
        train_new = train[:-step_in]
        test_new = train[-step_in:]

    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_sc = scaler.fit_transform(train_new)
    test_sc = scaler.transform(test_new)

    '''
    ANN model
    '''
    if model == 'ann':
        n_steps_in = step_in
        n_steps_out = step_out

        train_ann_sc_list = [i[0] for i in train_sc.tolist()]
        test_ann_sc_list = [i[0] for i in test_sc.tolist()]

        X_train, y_train = split_multi_step(train_ann_sc_list, n_steps_in, n_steps_out)

        X_test = test_sc.reshape((1, n_steps_in))
        y_test = np.array(test).reshape((1, n_steps_out))

        nn_model = Sequential()
        nn_model.add(Dense(neuron, input_dim=n_steps_in, activation='relu', kernel_initializer=init))
        nn_model.add(Dense(n_steps_out))
        nn_model.compile(loss='mean_squared_error', optimizer='adam')
        early_stop = EarlyStopping(monitor='loss', patience=2, verbose=1)
        nn_model.fit(X_train, y_train, epochs=epoch, batch_size=1, verbose=1, callbacks=[early_stop], shuffle=False)

        y_pred_test_nn = nn_model.predict(X_test)
        y_pred_test_nn_inver = scaler.inverse_transform(y_pred_test_nn)

        MAE = metrics.mean_absolute_error(y_test, y_pred_test_nn_inver)
        MSE = metrics.mean_squared_error(y_test, y_pred_test_nn_inver)
        RMSE = np.sqrt(metrics.mean_squared_error(y_test, y_pred_test_nn_inver))
        print('ANN MAE:', MAE)
        print('ANN MSE:', MSE)
        print('ANN RMSE:', RMSE)

        idx = test.index
        pred = y_pred_test_nn_inver.reshape(y_pred_test_nn_inver.shape[1], y_pred_test_nn_inver.shape[0]).tolist()
        forecast = pd.DataFrame(pred, index=idx, columns=['Forecast'])
        real = pd.concat([train, test])
        plot_result(forecast, real)

        return MAE, MSE, RMSE

    '''
    LSTM model
    '''
    if model == 'lstm':
        n_steps_in = step_in
        n_steps_out = step_out

        train_lstm_sc_list = [i[0] for i in train_sc.tolist()]
        X_train, y_train = split_multi_step(train_lstm_sc_list, n_steps_in, n_steps_out)

        X_train_lmse = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test_lmse = np.array([test_sc])

        lstm_model = Sequential()
        lstm_model.add(LSTM(neuron, input_shape=(n_steps_in, 1), activation='relu', kernel_initializer=init,
                            return_sequences=False))
        lstm_model.add(Dense(n_steps_out))
        lstm_model.compile(loss='mean_squared_error', optimizer='adam')
        early_stop = EarlyStopping(monitor='loss', patience=2, verbose=1)
        lstm_model.fit(X_train_lmse, y_train, epochs=epoch, batch_size=1, verbose=1, shuffle=False,
                       callbacks=[early_stop])

        y_pred_test_lstm = lstm_model.predict(X_test_lmse)
        y_pred_test_lstm_inver = scaler.inverse_transform(y_pred_test_lstm)

        y_test = np.array(test).reshape((1, n_steps_out))

        MAE = metrics.mean_absolute_error(y_test, y_pred_test_lstm_inver)
        MSE = metrics.mean_squared_error(y_test, y_pred_test_lstm_inver)
        RMSE = np.sqrt(metrics.mean_squared_error(y_test, y_pred_test_lstm_inver))

        print('LSTM MAE:', MAE)
        print('LSTM MSE:', MSE)
        print('LSTM RMSE:', RMSE)

        idx = test.index
        pred = y_pred_test_lstm_inver.reshape(y_pred_test_lstm_inver.shape[1], y_pred_test_lstm_inver.shape[0]).tolist()
        forecast = pd.DataFrame(pred, index=idx, columns=['Forecast'])
        real = pd.concat([train, test])
        plot_result(forecast, real)

        return MAE, MSE, RMSE
