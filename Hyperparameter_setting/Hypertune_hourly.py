from Hypertune import *
import warnings
warnings.filterwarnings("ignore")

# Please change the file location before running this code.
hourly = "C:/Users/Prin/Desktop/dissertation/Datashare/hourly_data.csv"
sch_hourly = "C:/Users/Prin/Desktop/dissertation/Datashare/sch_hourly_data.csv"

hourly_df = pd.read_csv(hourly).set_index('Hour')
hourly_df.index = pd.to_datetime(hourly_df.index)

# Splitting 2018 data
hourly_2018 = hourly_df['2018']

# Extract Term dates label
term_dates = hourly_df['Term_dates'].unique()
term_dates = ['Christmas vacation 1']  # , 'Private study 1', 'Mid-year assessment']

'''
Tunning ARIMA
'''
# evaluate parameters
p_values = [0, 1, 2, 3]
d_values = range(0, 3)
q_values = range(0, 3)

arima_hourly_tuned = []
arima_record = []

# Search the best Arima order for each term date
print('ARIMA model hyperparameter tuning is started!!')
for t in term_dates:
    print('Tunning ARIMA model for {} period...'.format(t))
    dataset = hourly_2018[hourly_2018.Term_dates == t]
    order, mse, report = evaluate_models(dataset.IN.values, p_values, d_values, q_values)
    print('')
    arima_hourly_tuned.append([t, order, mse])
    arima_record.append(report)

# Create dataframe to present the best order of each term date
arima_best = pd.DataFrame(arima_hourly_tuned, columns=['Term Date', 'ARIMA(p,d,q)', 'MSE']).set_index('Term Date')
print('Proper p,d,q of ARIMA of each term date')
# arima_best_config

# Store the ARIMA tuning record into dictionary
mse_report = {}
for k, v in zip(term_dates, arima_record):
    mse_report[k] = v

'''
Tuning ANN
'''
ann_hourly_tuned = []
report_en_ann = []
report_init_ann = []

# Search the best number of epoch and nuerons and initializer for each term date
print('ANN model hyperparameter tuning is started!!')
for t in term_dates:
    print('Tunning ANN model for {} period...'.format(t))
    dataset = hourly_2018[hourly_2018.Term_dates == t]
    inti, e, n, report_en, report_init = tune_nn(dataset, 'ann')
    print('')
    ann_hourly_tuned.append([t, inti, e, n])
    report_en_ann.append(report_en)
    report_init_ann.append(report_init)

# Create dataframe to present the best order of each term date
ann_best = pd.DataFrame(ann_hourly_tuned, columns=['Term Date', 'Initializer', 'epoch', 'nueron']).set_index(
    'Term Date')

# Store the ANN tuning record into dictionary
ann_en_report = {}
for k, v in zip(term_dates, report_en_ann):
    ann_en_report[k] = v

ann_init_report = {}
for k, v in zip(term_dates, report_init_ann):
    ann_init_report[k] = v
'''
Tuning LSTM
'''
lstm_hourly_tuned = []
report_en_lstm = []
report_init_lstm = []

# Search the best number of epoch and nuerons and initializer for each term date
print('LSTM model hyperparameter tuning is started!!')
for t in term_dates:
    print('Tunning LSTM model for {} period...'.format(t))
    dataset = hourly_2018[hourly_2018.Term_dates == t]
    inti, e, n, report_en, report_init = tune_nn(dataset, 'lstm')
    print('')
    lstm_hourly_tuned.append([t, inti, e, n])
    report_en_lstm.append(report_en)
    report_init_lstm.append(report_init)

# Create dataframe to present the best order of each term date
lstm_best = pd.DataFrame(lstm_hourly_tuned, columns=['Term Date', 'Initializer', 'epoch', 'nueron']).set_index(
    'Term Date')

# Store the LSTM tuning record into dictionary
lstm_en_report = {}
for k, v in zip(term_dates, report_en_lstm):
    lstm_en_report[k] = v

lstm_init_report = {}
for k, v in zip(term_dates, report_init_lstm):
    lstm_init_report[k] = v

'''
Export Best fit condition
'''
# Please change the exported location which is suit for your local drive
arima_best.to_csv('C:/Users/Prin/Desktop/dissertation/Datashare/arima_order.csv')
print('arima_order.csv is exported!!')
print('')

ann_best.to_csv('C:/Users/Prin/Desktop/dissertation/Datashare/ann_condition.csv')
print('ann_condition.csv is exported!!')
print('')

lstm_best.to_csv('C:/Users/Prin/Desktop/dissertation/Datashare/lstm_condition.csv')
print('lstm_condition.csv is exported!!')
print('')
