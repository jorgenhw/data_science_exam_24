import pandas as pd
import sys
from utils import *
import os
from neuralprophet import set_log_level
from hyperopt.pyll.base import scope
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

# Disable logging messages unless there is an error
set_log_level("ERROR")

train_size = ['small', 'large']
test_size = ['small', 'large']
forecast_horizons = [10, 50]

train_size = ['small']
test_size = ['small']
forecast_horizons = [50]

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
        #ar_layers = df[(df['train_size'] == train) & (df['forecast_horizon'] == forecast_horizon)]['ar_layers'].values[0]
        
        daily_seasonality = False
        weekly_seasonality = False
        n_forecasts = forecast_horizon                              # Number of steps ahead of prediction time step to forecast.   

        model = NeuralProphet(
            n_forecasts=n_forecasts,
            n_lags=n_lags,
            #ar_layers=ar_layers,
            yearly_seasonality=yearly_seasonality,
            seasonality_mode=seasonality_mode,
            n_changepoints=n_changepoints,
            epochs=epochs,
            daily_seasonality=daily_seasonality,
            weekly_seasonality=weekly_seasonality
        )

        model.fit(df_train, freq='MS')

        train_list, test_list = data_rolling_origin_prep(df_train, df_test, forecast_horizon)
        for i, (train_data, test_data) in enumerate(zip(train_list, test_list)):

            print(f'Iteration: {i}')

            future = model.make_future_dataframe(train_data, periods=forecast_horizon, n_historic_predictions=False)

            forecast = model.predict(future)

            # Remove NA values and gather forecasts from all columns
            forecasts = []
            for col in forecast.columns:
                if col.startswith('yhat'):
                    forecasts.extend(forecast.dropna(subset=[col])[col].values.tolist())

            # append metrics to list
            if 'forecast_df' not in locals():
                forecast_df = pd.DataFrame(columns=[f'forecast_{i}' for i in range(len(train_list))])
                # add ds column to forecast_df
                forecast_df['ds'] = df_test.index
                # add true values to forecast_df
                forecast_df['y'] = df_test['y'].values
            
            # add i number of NA values to start of forecast_mean and forecast_horizon - i after
            forecasts_na = np.concatenate([np.repeat(np.nan, i), forecasts, np.repeat(np.nan, len(train_list) - 1 - i)])
            forecast_df[f'forecast_{i}'] = forecasts_na

            # calculate metrics
            mse = mean_squared_error(test_data['y'], forecasts)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(test_data['y'], forecasts)
            mape = mean_absolute_percentage_error(test_data['y'], forecasts)
            r2 = r2_score(test_data['y'], forecasts)

            model_params = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'mape': mape,
                'r2': r2
            }

            hyperparameters_dict = {
                'train_size': train,
                'test_size': test,
                'forecast_horizon': forecast_horizon,
                'n_lags': n_lags,
                #'ar_layers': ar_layers,
                'n_changepoints': n_changepoints,
                'yearly_seasonality': yearly_seasonality,
                'seasonality_mode': seasonality_mode,
                'epochs': epochs,
                'iteration': i
            }

            # combine hyperparameters with metrics
            model_params.update(hyperparameters_dict)

            # add dict to all_metrics
            all_metrics.append(model_params)
        
                # create directory and save all outputs
        os.makedirs(f'NeuralProphet/outputs/test/Partition_horizon_{forecast_horizon}_train_{train}', exist_ok=True)

        # save forecast_df
        forecast_df.to_csv(f'NeuralProphet/outputs/test/Partition_horizon_{forecast_horizon}_train_{train}/forecast_df.csv', index=False)

        # remove forecast_df from locals
        del forecast_df

# merge all dictionaries in list
all_metrics_df = pd.DataFrame(all_metrics)

# create directory and save all outputs
os.makedirs(f'NeuralProphet/outputs/test', exist_ok=True)

# save all metrics
all_metrics_df.to_csv('NeuralProphet/outputs/test/all_metrics.csv', index=False)

# summarise bases on train and horizon both mean and std
all_metrics_df.groupby(['train_size', 'forecast_horizon']).agg({'rmse': ['mean', 'std'], 'mape': ['mean', 'std']}).to_csv('NeuralProphet/outputs/test/np_summary.csv')