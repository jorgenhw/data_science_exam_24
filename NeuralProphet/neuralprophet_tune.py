import pandas as pd
import sys
from utils import *
import os
from neuralprophet import set_log_level
from hyperopt.pyll.base import scope

# Disable logging messages unless there is an error
set_log_level("ERROR")

train_size = ['small', 'large']
test_size = ['small', 'large']
forecast_horizons = [10, 50]

# train_size = ['small']
# test_size = ['small']
f#orecast_horizons = [50]

all_metrics = []

i = 1

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

    # try:

        df_train, df_test = prepare_data_for_neural_prophet(train, test)

        if train == 'small':
            valid_size = forecast_horizon/100
        elif train == 'large':
            valid_size = forecast_horizon/1000

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
        best_results_df.to_csv(f'NeuralProphet/outputs/tuning/all_metrics_{i}.csv', index=False)

        i += 1

        # # create directory and save all outputs
        # os.makedirs(f'NeuralProphet/outputs/tuning/Partition_horizon_{forecast_horizon}_train_{train}_train', exist_ok=True)

        # # save model params
        # with open(f'NeuralProphet/outputs/tuning/Partition_horizon_{forecast_horizon}_train_{train}_train/model_params.txt', 'w') as f:
        #     f.write(str(model_params))

        # # save rmse_n list
        # with open(f'NeuralProphet/outputs/tuning/Partition_horizon_{forecast_horizon}_train_{train}_train/rmse_n.txt', 'w') as f:
        #     f.write(str(rmse_n))

        # # save mape_n list
        # with open(f'NeuralProphet/outputs/tuning/Partition_horizon_{forecast_horizon}_train_{train}_train/mape_n.txt', 'w') as f:
        #     f.write(str(mape_n))
    # except:
    #     print(f'Error in Partition_test_{test}_train_{train}_train')

    #     # create directory and save all outputs
    #     os.makedirs(f'NeuralProphet/outputs/Partition_test_{test}_train_{train}_train', exist_ok=True)

    #     # save error output
    #     with open(f'NeuralProphet/outputs/Partition_test_{test}_train_{train}_train/error.txt', 'w') as f:
    #         f.write('Error in NeuralProphet')

# merge all dictionaries in list
all_metrics_df = pd.DataFrame(all_metrics)

# create directory and save all outputs
os.makedirs(f'NeuralProphet/outputs/tuning', exist_ok=True)

# save all metrics
all_metrics_df.to_csv('NeuralProphet/outputs/tuning/all_metrics.csv', index=False)