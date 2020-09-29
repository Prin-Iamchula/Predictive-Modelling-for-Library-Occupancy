import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARIMA
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.layers import LSTM
import warnings
warnings.filterwarnings("ignore")  # Don't want to see the warnings in the notebook

'''
Hyperparameter tuning for ARIMA
'''
# evaluate an ARIMA model for a given order (p,d,q)
def evaluate_arima_model(X, arima_order):
    # prepare training dataset
    train_size = int(len(X) * 0.8)
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]
    # make predictions
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=arima_order)
        model_fit = model.fit(disp=0)
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])
    # calculate out of sample error
    error = mean_squared_error(test, predictions)
    return error

# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values):
    mse_report = [] 
    dataset = dataset.astype('float32')
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                try:
                    mse = evaluate_arima_model(dataset, order)
                    if mse < best_score:
                        best_score, best_cfg = mse, order
                    mse_report.append('ARIMA%s MSE=%.3f' % (order,mse))
                    # print('ARIMA%s MSE=%.3f' % (order,mse))
                except:
                    continue
    print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))
    return best_cfg,best_score,mse_report


'''
Hyperparameter tuning for ANN and LSTM
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
    plt.plot(pred, label='LSTM forecasting')
    plt.title(mod + "'s Prediction")
    plt.xlabel('Observation')
    plt.ylabel('Adj Close scaled')
    plt.legend()
    plt.show()


def ann_hypertune(X, step_in, model, epoch, neuron, init='he_uniform'):
    train_size = int(len(X) * 0.6)
    train, test = X[0:train_size], X[train_size:]

    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_sc = scaler.fit_transform(train)
    test_sc = scaler.transform(test)

    '''
    ANN model
    '''
    if model == 'ann':
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

        # plot_one(y_test_inver,y_pred_test_nn_inver,'ANN')

        return MAE, MSE, RMSE

    '''
    LSTM model
    '''
    if model == 'lstm':
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
        # plot_one(y_test_inver,y_pred_test_lstm_inver,'LSTM')

        return MAE, MSE, RMSE

def tune_nn(df,mod):
    result_en = []
    for e in [50,100,200,300]:
        for n in [50,100,200]:
            error=[]
            mae,mse,rmse = ann_hypertune(df[['IN']],step_in=1,model=mod,epoch=e,neuron=n)
            error=[e,n,1,mae,mse,rmse]
            # print(error)
            result_en.append(error)

    # Select best epoch and neuron
    best_e, best_n = 50, 50
    mse, rmse = float("inf"),float("inf")
    for i in result_en:   
        if i[4]<mse and i[5]<rmse:
            mse, rmse = i[4], i[5]
            best_e, best_n = i[0], i[1]

    # Tuning initializer
    result_init = init_tune(df, best_e, best_n, 1, mod)
    mse2, rmse2 = float("inf"),float("inf")
    for j in result_init:
        if j[2]<mse2 and j[3]<rmse2:
            mse2, rmse2 = j[2], j[3]
            best_init = j[0]

    return best_init, best_e, best_n, result_en, result_init

def visual_en(df):
    df = pd.DataFrame(df, columns=['Epoch','Neuron','n_step','MAE','MSE','RMSE'])
    return df.style.highlight_min(color = 'yellow')

def init_tune(df,e,n,s,mod):
    init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero',
              'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']

    result_nn = []
    for i in init_mode:
        error=[]
        mae,mse,rmse = ann_hypertune(df[['IN']],step_in=s,model=mod,epoch=e,neuron=n,init=i)
        error=[i,mae,mse,rmse]
        print(error)
        result_nn.append(error)

    return result_nn

def visual_init(alist):
    cols=['Initializer','MAE','MSE','RMSE']
    df = pd.DataFrame(alist, columns=cols)
    return df.style.highlight_min(color = 'green')