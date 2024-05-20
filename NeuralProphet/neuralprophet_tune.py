import pandas as pd
import sys
from utils import *
import os
from neuralprophet import set_log_level
from hyperopt.pyll.base import scope

# Disable logging messages unless there is an error
set_log_level("ERROR")

dataset = ['climate', 'weather']
train_size = ['small', 'large']
test_size = ['small', 'large']
forecast_horizons = [10, 50]

# train_size = ['small']
# test_size = ['small']
#forecast_horizons = [50]

all_metrics = []


for data in dataset:
    for train in train_size:
        for forecast_horizon in forecast_horizons:
            if forecast_horizon == 10:
                test = 'small'
                if train == 'small':
                    max_n_lags = 50
                elif train == 'large':
                    max_n_lags = 500
            elif forecast_horizon == 50:
                test = 'large'
                if train == 'small':
                    max_n_lags = 50 # cannot do proper hyper parameter tune for this, so we do no validation
                elif train == 'large':
                    max_n_lags = 500
            
            # print dashes
            print('-'*50)
            # print train forecast horizon
            print(f'Test: {test}',
                f'Train: {train}',
                    f'Forecast Horizon: {forecast_horizon}')
            # print dashes
            print('-'*50)

            df_train, df_test = prepare_data_for_neural_prophet(data, train, test)

            if data == 'climate':
                df_train, df_test = fix_dates_month(df_train, df_test)

            if train == 'small':
                valid_size = forecast_horizon/100
            elif train == 'large':
                valid_size = forecast_horizon/1000

            # df_train = make_series_stationary(df_train).fillna(0)

            # optimal_lags_dict = calculate_optimal_lags(df_train)   
            # optimal_lags = max(optimal_lags_dict.values())

            # df_train = calculate_moving_averages(df_train, 'M')

            # lagged_regressor_cols = df_train.columns[2:]

            epochs=[100]
            yearly_seasonality=[4,5,6,7] # ['True', 4,5,6,7]
            # loss_func=['MAE','MSE','Huber']
            seasonality_mode=['additive','multiplicative']
            #n_changepoints=[5,10,20,50]
            # learning_rate=[0.01,0.001,1]
            ar_layers = [[], [32], [32, 32], [32, 32, 32], [32, 32, 32, 32]]

            model_params =\
            {
            'epochs':hp.choice('epochs',epochs), 
            'yearly_seasonality':hp.choice('yearly_seasonality',yearly_seasonality),
            # 'loss_func':hp.choice('loss_func',loss_func),
            'seasonality_mode': hp.choice('seasonality_mode',seasonality_mode),     # additive = T+S+e, (Trend, Seasonality, error)
                                                                                    # multiplicative = T*S*e 
            'n_changepoints':scope.int(hp.quniform('n_changepoints', 0, 50, 5)),            # Number of potential trend changepoints to include
            'n_lags': scope.int(hp.quniform('n_lags', 0, max_n_lags, 5)),                        # previous time series steps to include in AR (or) AR-Order
            'ar_layers':hp.choice('ar_layers', ar_layers),                   # architecture layers for autoregression
            # 'learning_rate':hp.choice('learning_rate',learning_rate),               
            }

            ip_params=\
            {
            'df':df_train,                                        # dataframe
            'freq':'MS',                                    # model calculates frequency automatically
            'n_historic_predictions':True,                  # number of historic points included for past projection
            'periods':forecast_horizon,                                   # number of points for future projection
            'valid_p':valid_size,                                  # train_test_split
            'max_evals': 100,                                # maximum evaluations for hyperparameter tuning
            'lagged_regressor_cols': None, # columns used as lagged regressors
            }

            op_params=\
            {
            'daily_seasonality':False,
            'weekly_seasonality':False,
            'n_forecasts': forecast_horizon,                              # Number of steps ahead of prediction time step to forecast.   
            }

            if train == 'small':
                if forecast_horizon == 50:
                    # remove valid_p from dict
                    ip_params.pop('valid_p')
                    best_results = train_neural_prophet_noval(model_params, ip_params, op_params)
            else:
                best_results = train_neural_prophet(model_params, ip_params, op_params)

            best_results['ar_layers'] = ar_layers[best_results['ar_layers']]
            best_results['yearly_seasonality'] = yearly_seasonality[best_results['yearly_seasonality']]
            best_results['seasonality_mode'] = seasonality_mode[best_results['seasonality_mode']]
            best_results['epochs'] = epochs[best_results['epochs']]

            hyperparameters_dict = {
                            'dataset': data,
                            'train_size': train,
                            'forecast_horizon': forecast_horizon
                        }
            
            # combine hyperparameters with metrics
            best_results.update(hyperparameters_dict)

            all_metrics.append(best_results)

            # merge all dictionaries in list
            best_results_df = pd.DataFrame(best_results)

            # create directory and save all outputs
            os.makedirs(f'NeuralProphet/outputs/tuning', exist_ok=True)

            # save all metrics
            best_results_df.to_csv(f'NeuralProphet/outputs/tuning/all_metrics_{data}_{train}_{forecast_horizon}.csv', index=False)

# merge all dictionaries in list
all_metrics_df = pd.DataFrame(all_metrics)

# create directory and save all outputs
os.makedirs(f'NeuralProphet/outputs/tuning', exist_ok=True)

# save all metrics
all_metrics_df.to_csv('NeuralProphet/outputs/tuning/all_metrics.csv', index=False)