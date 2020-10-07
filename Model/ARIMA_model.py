import numpy as np
import itertools
import pandas as pd
import statsmodels.api as sm
from sklearn import metrics
from pmdarima import auto_arima
from statsmodels.tsa.arima_model import ARIMA
import warnings
import matplotlib.pyplot as plt
import matplotlib
from pylab import rcParams

plt.style.use('fivethirtyeight')
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'
warnings.filterwarnings("ignore")

'''
Single-step ARIMA
'''
# Please change the file location before running this code.
arima_best = "C:/Users/Prin/Desktop/dissertation/Datashare/arima_best.csv"


def arima_o(hist_data, termdate, col):
    checker = True
    try:
        arima_order = pd.read_csv(arima_best)
    except:
        checker = False
        print('Error: File "arima_best.csv" cannot be found.')
        print('Please check the file location.')

    if checker == False:
        print('Stop predicting.')
        return [], 0, 0, 0

    else:
        # Retrieve arima order
        orders = arima_order.loc[arima_order['Term Date'] == termdate, ['ARIMA(p,d,q)']].values
        order_temp = orders[0][0]
        order = (int(order_temp[1]), int(order_temp[4]), int(order_temp[7]))

        # Predict!!
        predictions, MAE, MSE, RMSE = arima_model(hist_data, termdate, col, order)
        return predictions, MAE, MSE, RMSE


def arima_model(train_hist, period, group, pmd):
    train_set = train_hist[train_hist.Term_dates == period]
    train_set = train_set[[group]]

    train_size = int(len(train_set) * 0.6)
    train, test = train_set[0:train_size], train_set[train_size:]

    history = [x for x in train_set.values]
    test_set = [i for i in test.values]
    predictions = []
    for t in range(len(test_set)):
        model = ARIMA(history, order=pmd)
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        #         print(output)
        yhat = output[0]
        if yhat < 0:
            yhat = 0
        else:
            pass
        predictions.append(yhat)
        obs = test_set[t]
        history.append(obs)
    #         print('predicted= %f, expected= %f' % (yhat, obs))
    #         print('t = ',t)

    MAE = metrics.mean_absolute_error(test_set, predictions)
    MSE = metrics.mean_squared_error(test_set, predictions)
    RMSE = np.sqrt(metrics.mean_squared_error(test_set, predictions))

    print(period, 'ARIMA MAE:', MAE, 'ARIMA MSE:', MSE, 'ARIMA RMSE:', RMSE)

    # plot
    plt.figure(figsize=(30, 6))
    plt.plot(test_set)
    plt.plot(predictions, color='red')
    plt.show()

    return predictions, MAE, MSE, RMSE


'''
Multi-step ARIMA (SARIMA)
'''


class arima:
    def __init__(self, traindata, testdata):
        self.traindata = traindata
        self.testdata = testdata

    def investigate(self, fre):
        #         self.traindata.plot(figsize=(14,4))

        rcParams['figure.figsize'] = 13, 8

        decomposition = sm.tsa.seasonal_decompose(self.traindata, model='additive', freq=fre)
        fig = decomposition.plot()
        plt.show()

    def tune(self, M):
        stepwise_fit = auto_arima(self.traindata, start_p=1, start_q=1,
                                  max_p=3, max_q=3, m=M,
                                  start_P=0, seasonal=True,
                                  d=None, D=1, trace=True,
                                  error_action='ignore',  # we don't want to know if an order does not work
                                  suppress_warnings=True,  # we don't want convergence warnings
                                  stepwise=True)
        return stepwise_fit.summary()

    def fit(self, pdq, pdqm):
        mod = sm.tsa.statespace.SARIMAX(self.traindata, order=pdq
                                        , seasonal_order=pdqm
                                        , enforce_stationarity=False
                                        , enforce_invertibility=False)
        self.model = mod.fit()
        print(self.model.summary().tables[1])

    def diagnos(self):
        self.model.plot_diagnostics(figsize=(16, 8))
        plt.show()

    def predict(self, start_pre, end_pre):
        pred = self.model.get_prediction(start=pd.to_datetime(start_pre), end=pd.to_datetime(end_pre), dynamic=False)
        pred_ci = pred.conf_int()
        self.prediction = pred.predicted_mean

        hist = pd.concat([self.traindata, self.testdata])
        ax = hist.plot(label='History')

        self.prediction.plot(ax=ax, label='Forecast', alpha=.7, figsize=(20, 7))
        ax.fill_between(pred_ci.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color='k', alpha=.2)
        ax.set_xlabel('Date')
        ax.set_ylabel('Total amount of user')
        plt.legend()
        plt.show()

    def report(self):
        y_forecasted = self.prediction
        y_truth = self.testdata

        MAE = metrics.mean_absolute_error(y_truth, y_forecasted)
        MSE = metrics.mean_squared_error(y_truth, y_forecasted)
        RMSE = np.sqrt(metrics.mean_squared_error(y_truth, y_forecasted))

        compare = pd.concat([y_truth, y_forecasted], axis=1)
        compare.columns = ['Actual', 'Forecast']

        print('MAE:', MAE)
        print('MSE:', MSE)
        print('RMSE:', RMSE)

