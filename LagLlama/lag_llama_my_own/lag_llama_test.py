# fine-tune hyperparameters for lag-llama model
# context length: suggested are 32, 64, 128, 256, 512, 1024
# learning rate: suggested are 0.01, 5*0.001, 0.001, 5*0.0001, 0.0001, 5*0.00001
# use validation split to early stop the model with patience of 50 epochs

# zero-shot hyperparameters for lag-llama model
# context length: suggested are 32, 64, 128, 356, 512, 1024 (until decrease in performance)
# Rope scaling: on or off. Maybe specific values can be tunes

import os
import pandas as pd
from gluonts.evaluation import Evaluator
import pickle
from utils.lag_utils import *
import numpy as np

datasets = ['climate', 'weather']
train_size = ['small', 'large']
test_size = ['small', 'large']
forecast_horizons = [10, 50]

# train_size = ['small']
# forecast_horizons = [50]

# load df with hyperparameters
df = pd.read_csv('outputs/tuning/best_hyperparameters.csv')

all_metrics = []

for data in datasets:
    for train in train_size:
        for forecast_horizon in forecast_horizons:
            if forecast_horizon == 10:
                test = 'small'
            elif forecast_horizon == 50:
                test = 'large'
            
            if data == 'weather':
                freq = 'M'
            elif data == 'climate':
                freq = 'H'
            
            # select hyperparameters from df
            rope_scaling = df[(df['train_size'] == train) & (df['forecast_horizon'] == forecast_horizon)]['rope_scaling'].values[0]
            context_length = df[(df['train_size'] == train) & (df['forecast_horizon'] == forecast_horizon)]['context_length'].values[0]


            train_, test_ = prepare_data_for_lag_llama(data, train, test)

            train_list, test_list = data_rolling_origin_prep(train_, test_, forecast_horizon)
            
            print(f'Dataset_{data}',
                    f'Partition_train_{train}',
                    f'Context Length: {context_length}',
                    f'Rope Scaling: {rope_scaling}',
                    f'Forecast Horizon: {forecast_horizon}')
                
            for i, (train_data, test_data) in enumerate(zip(train_list, test_list)):
                #try:
                print(f'Iteration {i}')
                # combine train_data and test_data
                train_data = pd.concat([train_data, test_data], axis = 0)

                # Prepare the data for deepAR format
                train_data_lds = to_deepar_format(train_data, freq)

                forecast, ts = get_lag_llama_predictions(train_data_lds, forecast_horizon, torch.device('cpu'), context_length, rope_scaling)

                evaluator = Evaluator(num_workers = None)
                agg_metrics, ts_metrics = evaluator(iter(ts), iter(forecast))

                hyperparameters_dict = {
                    'context_length': context_length,
                    'rope_scaling': rope_scaling,
                    'train_size': train,
                    'test_size': test,
                    'forecast_horizon': forecast_horizon
                }

                # combine hyperparameters with metrics
                agg_metrics.update(hyperparameters_dict)

                all_metrics.append(agg_metrics)

                # append metrics to list
                if 'forecast_df' not in locals():
                    forecast_df = pd.DataFrame(columns=[f'forecast_{i}' for i in range(len(train_list))])
                    # add ds column to forecast_df
                    forecast_df['ds'] = test_.index
                    # add true values to forecast_df
                    forecast_df['y'] = test_['y'].values
                
                forecast_mean = forecast[0].samples.mean(axis=0)
                # add i number of NA values to start of forecast_mean and forecast_horizon - i after
                forecast_mean = np.concatenate([np.repeat(np.nan, i), forecast_mean, np.repeat(np.nan, len(train_list) - 1 - i)])
                forecast_df[f'forecast_{i}'] = forecast_mean
        
                # except Exception as e:
                #     print(f'Error in Partition_horizon{forecast_horizon}_train_{train}_context_{context_length}_rope_{rope_scaling}_iteration_{i}')

                #     # create directory and save all outputs
                #     os.makedirs(f'outputs/test/Partition_horizon_{forecast_horizon}_train_{train}/error', exist_ok=True)

                #     # save error output
                #     with open(f'outputs/test/Partition_horizon_{forecast_horizon}_train_{train}/error/error_context_{context_length}_rope_{rope_scaling}_iteration_{i}.txt', 'w') as f:
                #         f.write(f'Error in LagLlama: {e}')
                
            # create directory and save all outputs
            os.makedirs(f'outputs/test/Partition_horizon_{forecast_horizon}_train_{train}_{data}', exist_ok=True)

            # save forecast_df
            forecast_df.to_csv(f'outputs/test/Partition_horizon_{forecast_horizon}_train_{train}_{data}/forecast_df.csv', index=False)

            # remove forecast_df from locals
            del forecast_df

# merge all dictionaries in list
all_metrics_df = pd.DataFrame(all_metrics)

# create directory and save all outputs
os.makedirs(f'outputs/test', exist_ok=True)

# save all metrics
all_metrics_df.to_csv('outputs/test/all_metrics.csv', index=False)

# group by context length and rope scaling and calculate mean and std
all_metrics_df.groupby(['context_length', 'rope_scaling', 'train_size', 'forecast_horizon', 'dataset']).agg({'MSE': ['mean', 'std'],
                                                                                                                'abs_error': ['mean', 'std'],
                                                                                                                'RMSE': ['mean', 'std'],
                                                                                                                'NRMSE': ['mean', 'std'],
                                                                                                                'MAPE': ['mean', 'std'],
                                                                                                                'sMAPE': ['mean', 'std'],
                                                                                                                'MASE': ['mean', 'std']}).to_csv('outputs/test/all_metrics_grouped.csv')
