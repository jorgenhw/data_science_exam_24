import pandas as pd
import sys
from utils import *
import os

train_size = [50, 250, 750]
test_size = [12, 120, 360]

train = train_size[int(sys.argv[1])]
test = test_size[int(sys.argv[2])]

df_train = pd.read_csv(f'data/climate/data_partitions/Partition_test_{test}_train_{train}_train.csv')
df_test = pd.read_csv(f'data/climate/data_partitions/Partition_test_{test}_train_{train}_test.csv')

# rename columns to fit neural prophet requirements
df_train.rename(columns={'date': 'ds', 'AMOC0': 'y'}, inplace=True)
df_test.rename(columns={'date': 'ds', 'AMOC0': 'y'}, inplace=True)

# remove columns that are not needed
df_train.drop(columns=['time', 'AMOC1', 'AMOC2', 'GM'], inplace=True)
df_test.drop(columns=['time', 'AMOC1', 'AMOC2', 'GM'], inplace=True)

df_train['ds'] = pd.to_datetime(df_train['ds'])
df_test['ds'] = pd.to_datetime(df_test['ds'])

# Function to find the closest first day of the month
def closest_first_day(date):
    first_day_this_month = date.replace(day=1)
    first_day_next_month = first_day_this_month + pd.offsets.MonthBegin(1)
    if date.day > 15:
        return first_day_next_month
    else:
        return first_day_this_month

# Apply the function to each date in the 'dates' column
df_train['ds'] =df_train['ds'].apply(closest_first_day)
df_test['ds'] =df_test['ds'].apply(closest_first_day)

test_horizon = len(df_test)
forecast_horizon = len(df_test)

# calculate size of validation set
# if len(df_train) < forecast_horizon:
#     valid_size = 0.2
# elif len(df_train) > forecast_horizon:
#     valid_size = forecast_horizon/len(df_train)
# else:
#     valid_size = 0.2

# valid_size = forecast_horizon/len(df_train)

# df_train = make_series_stationary(df_train).fillna(0)

# optimal_lags_dict = calculate_optimal_lags(df_train)   
# optimal_lags = max(optimal_lags_dict.values())

# df_train = calculate_moving_averages(df_train, 'M')

# lagged_regressor_cols = df_train.columns[2:]

epochs=[100]
daily_seasonality=['auto']
weekly_seasonality=['auto']
yearly_seasonality=['auto'] # ['True', 4,5,6,7]
# loss_func=['MAE','MSE','Huber']
seasonality_mode=['additive','multiplicative']
n_changepoints=[10,30,60]
# learning_rate=[0.01,0.001,1]

model_params =\
{
'epochs':hp.choice('epochs',epochs), 
'daily_seasonality':hp.choice('daily_seasonality',daily_seasonality),
'weekly_seasonality':hp.choice('weekly_seasonality',weekly_seasonality),
'yearly_seasonality':hp.choice('yearly_seasonality',yearly_seasonality),
# 'loss_func':hp.choice('loss_func',loss_func),
'seasonality_mode': hp.choice('seasonality_mode',seasonality_mode),     # additive = T+S+e, (Trend, Seasonality, error)
                                                                        # multiplicative = T*S*e 
'n_changepoints':hp.choice('n_changepoints',n_changepoints),            # Number of potential trend changepoints to include
# 'learning_rate':hp.choice('learning_rate',learning_rate),               
}

ip_params=\
{
'df':df_train,                                        # dataframe
'freq':'MS',                                    # model calculates frequency automatically
'n_historic_predictions':True,                  # number of historic points included for past projection
'periods':forecast_horizon,                                   # number of points for future projection
'valid_p':0.2,                                  # train_test_split
'max_evals': 10,                                # maximum evaluations for hyperparameter tuning
'lagged_regressor_cols': None, # columns used as lagged regressors
}

op_params=\
{
'n_lags': 5,                        # previous time series steps to include in AR (or) AR-Order
'n_forecasts': forecast_horizon,                              # Number of steps ahead of prediction time step to forecast.   
# 'ar_layers':[32, 32, 32, 32],                   # architecture layers for autoregression
}


model_params, rmse_n, mape_n = train_neural_prophet(df_train, df_test, model_params, ip_params, op_params, test_horizon)

# create directory and save all outputs
os.makedirs(f'NeuralProphet/outputs/Partition_test_{test}_train_{train}_train', exist_ok=True)

# save model params
with open(f'NeuralProphet/outputs/Partition_test_{test}_train_{train}_train/model_params.txt', 'w') as f:
    f.write(str(model_params))

# save rmse_n list
with open(f'NeuralProphet/outputs/Partition_test_{test}_train_{train}_train/rmse_n.txt', 'w') as f:
    f.write(str(rmse_n))

# save mape_n list
with open(f'NeuralProphet/outputs/Partition_test_{test}_train_{train}_train/mape_n.txt', 'w') as f:
    f.write(str(mape_n))