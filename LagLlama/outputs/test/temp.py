import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

datasets = ['weather', 'climate']
all_forecasts_50 = []
loop_parameters_50 = []
for data in datasets:
    for horizon in [50]:
        for train_size in ['small', 'large']:
            for file in os.listdir(f'LagLlama/lag-llama/outputs/test/Partition_horizon_{horizon}_train_{train_size}_{data}'):
                if file.endswith('.csv'):
                    all_forecasts_50.append(pd.read_csv(f'LagLlama/lag-llama/outputs/test/Partition_horizon_{horizon}_train_{train_size}_{data}/' + file))
                    loop_parameters_50.append({'data': data, 'horizon': horizon, 'train_size': train_size})

all_forecasts_10 = []
loop_parameters_10 = []
for data in datasets:
    for horizon in [10]:
        for train_size in ['small', 'large']:
            for file in os.listdir(f'LagLlama/lag-llama/outputs/test/Partition_horizon_{horizon}_train_{train_size}_{data}'):
                if file.endswith('.csv'):
                    all_forecasts_10.append(pd.read_csv(f'LagLlama/lag-llama/outputs/test/Partition_horizon_{horizon}_train_{train_size}_{data}/' + file))
                    loop_parameters_10.append({'data': data, 'horizon': horizon, 'train_size': train_size})

# calculate the rmse for each column named forecast_i in each dataframe and ignore the na values. The true values are in the column y
all_rmse_50 = pd.DataFrame()
for idx, forecast in enumerate(all_forecasts_50):
    for i in range(len(forecast.columns) - 2):
        rmse = np.sqrt(np.mean((forecast['y'] - forecast[f'forecast_{i}'])**2))
        all_rmse_50.loc[idx, f'forecast_{i}'] = rmse

all_rmse_10 = pd.DataFrame()
for idx, forecast in enumerate(all_forecasts_10):
    for i in range(len(forecast.columns) - 2):
        rmse = np.sqrt(np.mean((forecast['y'] - forecast[f'forecast_{i}'])**2))
        all_rmse_10.loc[idx, f'forecast_{i}'] = rmse

# calculate mean and standard deviation of each row and save in new df
all_rmse_50['mean'] = all_rmse_50.mean(axis=1)
all_rmse_50['std'] = all_rmse_50.std(axis=1)
all_rmse_50['dataset'] = [loop_parameters_50[idx]['data'] for idx in range(len(loop_parameters_50))]
all_rmse_50['horizon'] = [loop_parameters_50[idx]['horizon'] for idx in range(len(loop_parameters_50))]
all_rmse_50['train_size'] = [loop_parameters_50[idx]['train_size'] for idx in range(len(loop_parameters_50))]

all_rmse_10['mean'] = all_rmse_10.mean(axis=1)
all_rmse_10['std'] = all_rmse_10.std(axis=1)
all_rmse_10['dataset'] = [loop_parameters_10[idx]['data'] for idx in range(len(loop_parameters_10))]
all_rmse_10['horizon'] = [loop_parameters_10[idx]['horizon'] for idx in range(len(loop_parameters_10))]
all_rmse_10['train_size'] = [loop_parameters_10[idx]['train_size'] for idx in range(len(loop_parameters_10))]

# concat only the mean values and std values
all_rmse = pd.concat([all_rmse_10[['mean', 'std', 'dataset', 'horizon', 'train_size']], all_rmse_50[['mean', 'std', 'dataset', 'horizon', 'train_size']]], axis=0)
all_rmse.to_csv('LagLlama/lag-llama/outputs/test/all_rmse.csv', index=False)
