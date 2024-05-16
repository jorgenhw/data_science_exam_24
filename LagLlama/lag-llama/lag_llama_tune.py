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

train_size = [50, 250, 750]
test_size = [12, 120, 360]

train_size = [50]
test_size = [12]
forecast_horizons = [6, 12]
rope_scaling = [True, False]

all_metrics = []

for train in train_size:
    for test in test_size:
        for forecast_horizon in forecast_horizons:
            if train == 50:
                context_length_list = [32, 50]
            elif train == 250:
                context_length_list = [32, 64, 128, 250]
            elif train == 750:
                context_length_list = [32, 64, 128, 256, 512, 750]

            train_, test_ = prepare_data_for_lag_llama(train, test)

            train_list, test_list = data_rolling_origin_prep(train_, test_, forecast_horizon)
            for rope in rope_scaling:
                for context_length in context_length_list:
                    print(f'Partition_test_{test}_train_{train}_train',
                          f'Context Length: {context_length}',
                          f'Rope Scaling: {rope}',
                          f'Forecast Horizon: {forecast_horizon}')
                    
                    for i, (train_data, test_data) in enumerate(zip(train_list, test_list)):
                        try:
                            # Prepare the data for deepAR format
                            train_data_lds = to_deepar_format(train_data, 'M')
                            test_data_lds = to_deepar_format(test_data, 'M')

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

                            # append metrics to list
                            all_metrics.append(agg_metrics)

                            # # create directory and save all outputs
                            # os.makedirs(f'outputs/Partition_test_{test}_train_{train}/metrics', exist_ok=True)
                            # os.makedirs(f'outputs/Partition_test_{test}_train_{train}/ts_metrics', exist_ok=True)

                            # # save metrics output
                            # with open(f'outputs/Partition_test_{test}_train_{train}/metrics/metrics_context_{context_length}_rope_{rope}_iteration_{i}.pkl', 'wb') as f:
                            #     pickle.dump(agg_metrics, f)
                            
                            # # save time series metrics output (it is a pandas dataframe)
                            # ts_metrics.to_csv(f'outputs/Partition_test_{test}_train_{train}/ts_metrics/ts_metrics_context_{context_length}_rope_{rope}_iteration_{i}.csv', index=False)
                
                        except:
                            print(f'Error in Partition_test_{test}_train_{train}_context_{context_length}_rope_{rope}_forecast_horizon_{forecast_horizon}_iteration_{i}')

                            # create directory and save all outputs
                            os.makedirs(f'outputs/Partition_test_{test}_train_{train}/error', exist_ok=True)

                            # save error output
                            with open(f'outputs/Partition_test_{test}_train_{train}/error/error_context_{context_length}_rope_{rope}_forecast_horizon_{forecast_horizon}_iteration_{i}.txt', 'w') as f:
                                f.write(f'Error in LagLlama')

# merge all dictionaries in list
all_metrics_df = pd.DataFrame(all_metrics)

# save all metrics
all_metrics_df.to_csv('outputs/all_metrics.csv', index=False)

# group by context length and rope scaling and calculate mean and std
all_metrics_df.groupby(['context_length', 'rope_scaling', 'train_size', 'test_size', 'forecast_horizon']).agg({'MSE': ['mean', 'std'],
                                                                                                                'abs_error': ['mean', 'std'],
                                                                                                                'RMSE': ['mean', 'std'],
                                                                                                                'NRMSE': ['mean', 'std'],
                                                                                                                'MAPE': ['mean', 'std'],
                                                                                                                'sMAPE': ['mean', 'std'],
                                                                                                                'MASE': ['mean', 'std']}).to_csv('outputs/all_metrics_grouped.csv')
