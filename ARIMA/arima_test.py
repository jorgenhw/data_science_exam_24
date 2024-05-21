from pmdarima import arima
from utils import *
import os
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import ast

datasets = ['climate','weather']
train_size = ['small', 'large']
test_size = ['small', 'large']
forecast_horizons = [10, 50]

# train_size = ['small']
# test_size = ['small']
# forecast_horizons = [10]

all_metrics = []

# read hyperparameters from csv
hyperparameters = pd.read_csv('ARIMA/outputs/tuning/arima_hyperparameters.csv')

for data in datasets:
    for train in train_size:
        for forecast_horizon in forecast_horizons:
            if forecast_horizon == 10:
                test = 'small'
                # if train == 'small':
                #     max_n_lags = 50
                # elif train == 'large':
                #     max_n_lags = 500
            elif forecast_horizon == 50:
                test = 'large'
                # if train == 'small':
                #     max_n_lags = 50
                # elif train == 'large':
                #     max_n_lags = 500

            # print dashes
            print('-'*50)
            # print train forecast horizon
            print(f'Test: {test}',
                f'Train: {train}',
                    f'Forecast Horizon: {forecast_horizon}')
            # print dashes
            print('-'*50)

            df_train, df_test = prepare_data_for_arima(data, train, test)
            
            if data == 'climate':
                df_train, df_test = fix_dates_month(df_train, df_test)
                m = 12
            elif data == 'weather':
                m = 24

            order = hyperparameters[(hyperparameters['train_size'] == train) & (hyperparameters['forecast_horizon'] == forecast_horizon)]['order'].values[0]
            seasonal_order = hyperparameters[(hyperparameters['train_size'] == train) & (hyperparameters['forecast_horizon'] == forecast_horizon)]['seasonal_order'].values[0]

            order = ast.literal_eval(order)
            seasonal_order = ast.literal_eval(seasonal_order)

            # # Fit the ARIMA model using the loaded parameters
            # model = ARIMA(df_train['y'], order=order, seasonal_order=seasonal_order).fit()

            # prepare data for rolling origin evaluation
            train_list, test_list = data_rolling_origin_prep(df_train, df_test, forecast_horizon)

            # make predictions
            for i, (train_data, test_data) in enumerate(zip(train_list, test_list)):

                # reset index
                train_data.reset_index(drop=True, inplace=True)

                # Fit the ARIMA model using the loaded parameters
                model = ARIMA(train_data['y'], order=order, seasonal_order=seasonal_order).fit(method_kwargs={'maxiter':300})

                forecasts = model.forecast(steps=forecast_horizon)

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
                    'dataset': data,
                    'train_size': train,
                    'test_size': test,
                    'forecast_horizon': forecast_horizon,
                    'order': order,
                    'seasonal_order': seasonal_order,
                    'iteration': i
                }

                # combine hyperparameters with metrics
                model_params.update(hyperparameters_dict)

                # add dict to all_metrics
                all_metrics.append(model_params)
            
                    # create directory and save all outputs
            os.makedirs(f'ARIMA/outputs/test/Partition_horizon_{forecast_horizon}_train_{train}_{data}', exist_ok=True)

            # save forecast_df
            forecast_df.to_csv(f'ARIMA/outputs/test/Partition_horizon_{forecast_horizon}_train_{train}_{data}/forecast_df.csv', index=False)

            # remove forecast_df from locals
            del forecast_df

# make all_metrics into a pandas dataframe
all_metrics_df = pd.DataFrame(all_metrics)

# create directory and save all outputs
os.makedirs(f'ARIMA/outputs/test', exist_ok=True)

# save all_metrics_df to csv
all_metrics_df.to_csv('ARIMA/outputs/test/arima_results.csv', index=False)

# summarise bases on train and horizon both mean and std
all_metrics_df.groupby(['dataset', 'train_size', 'forecast_horizon']).agg({'mae': ['mean', 'std'], 'mse': ['mean', 'std'], 'rmse': ['mean', 'std'], 'mape': ['mean', 'std']}).to_csv('ARIMA/outputs/test/arima_results_summary.csv')