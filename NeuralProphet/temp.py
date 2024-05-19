# create a pandas with column train, forecast_horizon, yearly_seasonality, seasonality_mode, n_changespoints, n_lags, ar_layers

import pandas as pd
import numpy as np
import os

# create a pandas with column train, forecast_horizon, yearly_seasonality, seasonality_mode, n_changespoints, n_lags, ar_layers
train = ['small']
forecast_horizon = [50]
yearly_seasonality = [True]
seasonality_mode = ['additive']
n_changepoints = [50]
n_lags = [50]
#ar_layers = [50]

# create a pandas with column train, forecast_horizon, yearly_seasonality, seasonality_mode, n_changespoints, n_lags, ar_layers
df = pd.DataFrame({
    'train_size': train,
    'forecast_horizon': forecast_horizon,
    'yearly_seasonality': yearly_seasonality,
    'seasonality_mode': seasonality_mode,
    'n_changepoints': n_changepoints,
    'n_lags': n_lags,
    #'ar_layers': ar_layers
})

# create directory and save all outputs
os.makedirs(f'NeuralProphet/outputs/tuning', exist_ok=True)

# save all metrics

df.to_csv('NeuralProphet/outputs/tuning/best_hyperparameters.csv', index=False)
