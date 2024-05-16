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

train_size = [100, 1000]
test_size = [50, 250]
forecast_horizons = [10, 50]

train_size = [50, 250]
test_size = [12]
forecast_horizons = [6]

# I decide to use full training as context and rope_scaling = True for train_size = 100
rope_scaling = [True]

all_metrics = []

for train in train_size:
    for test in test_size:
        for forecast_horizon in forecast_horizons:
            # if train == 100:
            #     continue
            # else:
            #     context_length_list = [32, 64, 128, 256, 512, 950]
            if train == 50:
                continue
            elif train == 250:
                context_length_list = [32, 64]
            elif train == 750:
                context_length_list = [32, 64, 128, 256, 512, 750]

            train_, test_ = prepare_data_for_lag_llama(train, test)

            for rope in rope_scaling:
                for context_length in context_length_list:
                    print(f'Partition_test_{test}_train_{train}_train',
                          f'Context Length: {context_length}',
                          f'Rope Scaling: {rope}',
                          f'Forecast Horizon: {forecast_horizon}')
                    
                    try:
                        # combine train_data and test_data
                        train_data = train_

                        # Prepare the data for deepAR format
                        train_data_lds = to_deepar_format(train_data, 'M')

                        forecast, ts = get_lag_llama_predictions(train_data_lds, forecast_horizon, torch.device('cpu'), context_length, rope)

                        evaluator = Evaluator(num_workers = None)
                        agg_metrics, ts_metrics = evaluator(iter(ts), iter(forecast))

                        hyperparameters_dict = {
                            'context_length': context_length,
                            'rope_scaling': rope,
                            'train_size': train,
                            'test_size': test,
                            'forecast_horizon': forecast_horizon
                        }

                        # combine hyperparameters with metrics
                        agg_metrics.update(hyperparameters_dict)

                        all_metrics.append(agg_metrics)

                    except:
                        print(f'Error in Partition_horizon{forecast_horizon}_train_{train}_context_{context_length}_rope_{rope}_iteration_{i}')

                        # create directory and save all outputs
                        os.makedirs(f'outputs/tuning/Partition_test_{test}_train_{train}/error', exist_ok=True)

                        # save error output
                        with open(f'outputs/tuning/Partition_test_{test}_train_{train}/error/error_context_{context_length}_rope_{rope}_iteration_{i}.txt', 'w') as f:
                            f.write(f'Error in LagLlama')
                
# merge all dictionaries in list
all_metrics_df = pd.DataFrame(all_metrics)

# create directory and save all outputs
os.makedirs(f'outputs/tuning', exist_ok=True)

# save all metrics
all_metrics_df.to_csv('outputs/tuning/all_metrics.csv', index=False)

# group by context length and rope scaling and calculate mean and std
all_metrics_df.groupby(['context_length', 'rope_scaling', 'train_size', 'test_size', 'forecast_horizon']).agg({'MSE': ['mean', 'std'],
                                                                                                                'abs_error': ['mean', 'std'],
                                                                                                                'RMSE': ['mean', 'std'],
                                                                                                                'NRMSE': ['mean', 'std'],
                                                                                                                'MAPE': ['mean', 'std'],
                                                                                                                'sMAPE': ['mean', 'std'],
                                                                                                                'MASE': ['mean', 'std']}).to_csv('outputs/tuning/all_metrics_grouped.csv')

# save best hyperparameters for each combination of train_size, test_size and forecast_horizon
best_hyperparameters = all_metrics_df.groupby(['train_size', 'test_size', 'forecast_horizon']).apply(lambda x: x.nsmallest(1, 'MSE')).reset_index(drop=True)[['train_size', 'test_size', 'forecast_horizon', 'context_length', 'rope_scaling']]

# add row to best_hyperparameters
best_hyperparameters.loc[len(best_hyperparameters)] = [100, 10, 10, 100, True]
best_hyperparameters.loc[len(best_hyperparameters)] = [100, 50, 50, 100, True]

# save best hyperparameters
best_hyperparameters.to_csv('outputs/tuning/best_hyperparameters.csv', index=False)