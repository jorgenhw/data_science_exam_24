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
#forecast_horizons = [50]

all_metrics = []

# load df with hyperparameters
df = pd.read_csv('NeuralProphet/outputs/tuning/best_hyperparameters.csv')

for train in train_size:
    for forecast_horizon in forecast_horizons:
        if forecast_horizon == 10:
            test = 'small'
        elif forecast_horizon == 50:
            test = 'large'
        
        # print dashes
        print('-'*50)
        # print train forecast horizon
        print(f'Test: {test}',
              f'Train: {train}',
                f'Forecast Horizon: {forecast_horizon}')
        # print dashes
        print('-'*50)

        df_train, df_test = prepare_data_for_neural_prophet(train, test)

        if train == 'small':
            valid_size = forecast_horizon/100
        elif train == 'large':
            valid_size = forecast_horizon/1000
        
        # select hyperparameters from df
        epochs=100
        yearly_seasonality=df[(df['train_size'] == train) & (df['forecast_horizon'] == forecast_horizon)]['yearly_seasonality'].values[0]
        seasonality_mode=df[(df['train_size'] == train) & (df['forecast_horizon'] == forecast_horizon)]['seasonality_mode'].values[0]
        n_changepoints=df[(df['train_size'] == train) & (df['forecast_horizon'] == forecast_horizon)]['n_changepoints'].values[0]
        n_lags=df[(df['train_size'] == train) & (df['forecast_horizon'] == forecast_horizon)]['n_lags'].values[0]
        ar_layers = df[(df['train_size'] == train) & (df['forecast_horizon'] == forecast_horizon)]['ar_layers'].values[0]

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