import pandas as pd
import numpy as np
import sys
from neuralprophet import NeuralProphet
from hyperopt import hp, fmin, tpe, Trials,STATUS_OK 
from hyperopt.pyll.base import scope
from hyperopt.early_stop import no_progress_loss
import torch
from utils import *

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

df_train = make_series_stationary(df_train).fillna(0)

optimal_lags_dict = calculate_optimal_lags(df_train)   
optimal_lags = max(optimal_lags_dict.values())

df_train = calculate_moving_averages(df_train, 'M')

lagged_regressor_cols = df_train.columns[2:]

epochs=[200]
daily_seasonality=['auto']
weekly_seasonality=['auto']
yearly_seasonality=['auto']
# loss_func=['MAE','MSE','Huber']
seasonality_mode=['additive','multiplicative']
n_changepoints=[30,60]
# learning_rate=[0.01,0.001,1]

model_params =\
{
'epochs':hp.choice('epochs',epochs), 
'daily_seasonality':hp.choice('daily_seasonality',daily_seasonality),
'weekly_seasonality':hp.choice('weekly_seasonality',weekly_seasonality),
'yearly_seasonality':hp.choice('yearly_seasonality',yearly_seasonality),
'loss_func':hp.choice('loss_func',loss_func),
'seasonality_mode': hp.choice('seasonality_mode',seasonality_mode),     # additive = T+S+e, (Trend, Seasonality, error)
                                                                        # multiplicative = T*S*e 
'n_changepoints':hp.choice('n_changepoints',n_changepoints),            # Number of potential trend changepoints to include
'learning_rate':hp.choice('learning_rate',learning_rate),               
}

ip_params=\
{
'df':df_train,                                        # dataframe
'freq':None,                                    # model calculates frequency automatically
'n_historic_predictions':True,                  # number of historic points included for past projection
'periods':100,                                   # number of points for future projection
'valid_p':0.2,                                  # train_test_split
'max_evals': 10,                                # maximum evaluations for hyperparameter tuning
'lagged_regressor_cols': lagged_regressor_cols, # columns used as lagged regressors
}

op_params=\
{
'n_lags': optimal_lags ,                        # previous time series steps to include in AR (or) AR-Order
'n_forecasts': 100,                              # Number of steps ahead of prediction time step to forecast.   
'ar_layers':[32, 32, 32, 32],                   # architecture layers for autoregression
}
