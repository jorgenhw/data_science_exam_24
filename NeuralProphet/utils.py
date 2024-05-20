from statsmodels.tsa.stattools import adfuller
from neuralprophet import NeuralProphet
from hyperopt import hp, fmin, tpe, Trials,STATUS_OK 
from hyperopt.pyll.base import scope
from hyperopt.early_stop import no_progress_loss
import torch
import numpy as np
import pandas as pd
from typing import List, Tuple

def data_rolling_origin_prep(data_train: pd.DataFrame, data_test: pd.DataFrame, horizon: int) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
    """
    Prepare data for rolling origin validation

    Parameters
    ----------
    data_train : pd.DataFrame
        Training data
    data_test : pd.DataFrame
        Testing data
    horizon : int
        Forecast horizon
    
    Returns
    -------
    X : List[pd.DataFrame]
        List of training data for each iteration
    y : List[pd.DataFrame]
        List of testing data for each iteration
    """
    X, y = [], []
    for i in range(len(data_test) - horizon + 1):
        X.append(pd.concat([data_train, data_test[0:i]], axis=0))
        y.append(data_test[i:i + horizon])
    return X, y

def prepare_data_for_neural_prophet(data, train, test):
    df_train = pd.read_csv(f'data/{data}/splits/train/train_{train}.csv')
    df_test = pd.read_csv(f'data/{data}/splits/test/test_{test}_for_train_{train}.csv')

    # rename columns to fit neural prophet requirements
    df_train.rename(columns={'date': 'ds', 'AMOC0': 'y'}, inplace=True)
    df_test.rename(columns={'date': 'ds', 'AMOC0': 'y'}, inplace=True)

    # remove columns that are not needed
    df_train.drop(columns=['time', 'AMOC1', 'AMOC2', 'GM'], inplace=True)
    df_test.drop(columns=['time', 'AMOC1', 'AMOC2', 'GM'], inplace=True)

    df_train['ds'] = pd.to_datetime(df_train['ds'])
    df_test['ds'] = pd.to_datetime(df_test['ds'])

    # df_train.set_index('ds', inplace=True)
    # df_test.set_index('ds', inplace=True)
    
    return df_train, df_test

def fix_dates_month(df_train, df_test):
        # Function to find the closest first day of the month
    def closest_first_day(date):
        first_day_this_month = date.replace(day=1)
        first_day_next_month = first_day_this_month + pd.offsets.MonthBegin(1)
        if date.day > 15:
            return first_day_next_month
        else:
            return first_day_this_month

    # Apply the function to each date in the 'dates' column
    df_train['ds'] =df_train['ds'].apply(closest_first_day)
    df_test['ds'] =df_test['ds'].apply(closest_first_day)

    return df_train, df_test

def apply_differencing(series, max_diff, alpha):
    # H0: Time-Series is Non-Stationary
    # H1: Time-Series is Stationary
    
    # if p_value <= alpha: # reject null hypothesis (stationary)
    # else: # fail to reject null hypothesis (non-stationary)        
    
    num_diff = 0
    result = adfuller(series.dropna(), autolag='AIC')
    p_value = result[1]

    while p_value > alpha and num_diff < max_diff: # while non-stationary, keep differencing the series to make it stationary
        series = series.diff().dropna()
        result = adfuller(series, autolag='AIC')
        p_value = result[1]
        num_diff += 1

    return num_diff, p_value

def make_series_stationary(df, max_diff=10, alpha=0.05):
    if 'ID' in df.columns:
        for id_value in df['ID'].unique():
            series = df[df['ID'] == id_value]['y']
            num_diff, p_value = apply_differencing(series, max_diff, alpha)
            print(f"ID {id_value}: Series is {'stationary' if p_value <= alpha else 'non-stationary'} \
                                                                after {num_diff} differencing operation(s).")
            if num_diff > 0:
                df.loc[df['ID'] == id_value, 'I'] = series.diff(periods=num_diff).fillna(0)
    else:
        series = df['y']
        num_diff, p_value = apply_differencing(series, max_diff, alpha)
        print(f"Series is {'stationary' if p_value <= alpha else 'non-stationary'} after {num_diff} differencing operation(s).")
        if num_diff > 0:
            df['I'] = series.diff(periods=num_diff).fillna(0)

    return df


import numpy as np
from statsmodels.tsa.stattools import acf

# it outputs 2 and 19 as optimal lags soooo we search as a hyperparameter

# def find_optimal_lags(series, alpha):
#     autocorr, confint = acf(series.dropna(), alpha=alpha, nlags=100, fft=True)
#     conf_offset = confint[:, 1] - autocorr
#     optimal_lags = np.where((autocorr < conf_offset) & (autocorr > -conf_offset))[0]

#     if len(optimal_lags) == 0:
#         return 0
#     else:
#         return optimal_lags[0] - 1

# def calculate_optimal_lags(df, alpha=0.05):
#     optimal_lags_dict = {}

#     series = df['y']
#     optimal_lags_dict['1'] = find_optimal_lags(series, alpha)

#     return optimal_lags_dict

def calculate_moving_averages(df, freq=None):
    # Mapping common frequencies to moving average intervals
    intervals = {
        'D': [7, 14, 30, 60, 90, 180, 365],  # Daily data: week, fortnight, month, 2-months, quarter, half-year, year
        'W': [4, 8, 13, 26, 52],             # Weekly data: month, 2-months, quarter, half-year, year
        'M': [1, 3, 6, 12, 24],              # Monthly data: month, quarter, half-year, year, 2-years
        'H': [24, 72, 168, 336, 720],        # Hourly data: day, 3-days, week, 2-weeks, month
        'T': [15, 30, 60, 120, 240, 720, 1440] # Minutely data: quarter-hour, half-hour, hour, 2-hours, 4-hours, 12-hours, day
    }
                
    if freq:
        selected_intervals = intervals.get(freq)
        for interval in selected_intervals:
            column_name = f'MA_{interval}'
            if 'ID' in df.columns:
                # Calculate moving average per ID and backfill within each ID
                df[column_name] = df.groupby('ID')['y'].transform(lambda x: x.rolling(window=interval).mean()\
                                                                                      .fillna(method='bfill'))
            else:
                # Calculate moving average for entire series and backfill
                df[column_name] = df['y'].rolling(window=interval).mean().fillna(method='bfill')            
    else:
        print("Provided frequency is not recognized. Unable to calculate moving averages.")

    return df

def rolling_origin_func(df, forecast_horizon, roll_step=1):
    '''
    Rolling Origin Validation for Time Series Forecasting
    Parameters:
    df (DataFrame): Time Series Data
    forecast_horizon (int): Number of periods to forecast
    roll_step (int): Step size for rolling origin validation
    
    Returns:
    train_test_splits (list): List of tuples containing train and test dataframes
    
    Source: https://github.com/michael-berk/DS_academic_papers/blob/master/28_prophet_vs_neural_prophet.py
    '''

    train_test_split_indices = list(range(len(df.index) - forecast_horizon - 10, len(df.index) - forecast_horizon, roll_step))
    train_test_splits = [(df.iloc[:i, :], df.iloc[i:(i+forecast_horizon), :]) for i in train_test_split_indices]
    return train_test_splits

def accuracy(obs, pred):
    """
    Calculate accuracy measures

    :param obs: pd.Series of observed values
    :param pred: pd.Series of forecasted values
    :return: dict with accuracy measures
    """

    obs, pred = np.array(obs.dropna()), np.array(pred.dropna())

    assert len(obs) == len(pred), f'accuracy(): obs len is {len(obs)} but preds len is {len(pred)}'

    rmse = np.sqrt(np.mean((obs - pred)**2))
    mape = np.mean(np.abs((obs - pred) / obs)) 

    return (rmse, mape)

def train_neural_prophet(model_params, ip_params, op_params):
        
    # Combine model parameters & additional input & output parameters
    args = {'model_params':model_params,'ip_params':ip_params,'op_params':op_params} 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # GPU Check        
    print('Device Used: {}'.format(device))
    trainer_config = {"accelerator":"cuda"} # use GPU if available, no need for model.to(device) for neuralprophet           
    
    def optimize(args): # Hyperparameter tuning with Hyperopt
        df = args['ip_params']['df']        
        model = NeuralProphet( **{**args['model_params'],**args['op_params']})        
        if args['ip_params']['lagged_regressor_cols'] is not None:
            if args['op_params']['n_lags']>0:
                for col in args['ip_params']['lagged_regressor_cols']:
                    model = model.add_lagged_regressor(col, normalize="standardize")
            else:
                df = df[list(set(df.columns) - set(ip_params['lagged_regressor_cols']))]
        try:
            df_train, df_val = model.split_df(df, freq=args['ip_params']['freq'], valid_p=args['ip_params']['valid_p'])
            train_metrics = model.fit(df_train, freq=args['ip_params']['freq'], validation_df=df_val)
            test_metrics = model.test(df_val)
        
            return {'loss':test_metrics['RMSE_val'].reset_index(drop=True)[0], 'status': STATUS_OK }
        except Exception as e:
            print(f"Exception in objective function: {e}")
            return {'loss': 1000, 'status': 'fail'}

    early_stop_fn = no_progress_loss(iteration_stop_count=int((args['ip_params']['max_evals'])*0.2), percent_increase=5)
    trials = Trials() # if you want more info on the hyperopt search, you use this object afterwards, e.g. trials.trials
    best_results = fmin(optimize, space=args, algo=tpe.suggest, trials=trials, max_evals=args['ip_params']['max_evals'], early_stop_fn = early_stop_fn, show_progressbar=False)

    return best_results

def train_neural_prophet_noval(model_params, ip_params, op_params):
        
    # Combine model parameters & additional input & output parameters
    args = {'model_params':model_params,'ip_params':ip_params,'op_params':op_params} 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # GPU Check        
    print('Device Used: {}'.format(device))
    trainer_config = {"accelerator":"cuda"} # use GPU if available, no need for model.to(device) for neuralprophet           
    
    def optimize(args): # Hyperparameter tuning with Hyperopt
        df = args['ip_params']['df']        
        model = NeuralProphet( **{**args['model_params'],**args['op_params']})        
        if args['ip_params']['lagged_regressor_cols'] is not None:
            if args['op_params']['n_lags']>0:
                for col in args['ip_params']['lagged_regressor_cols']:
                    model = model.add_lagged_regressor(col, normalize="standardize")
            else:
                df = df[list(set(df.columns) - set(ip_params['lagged_regressor_cols']))]
        try:
            train_metrics = model.fit(df, freq=args['ip_params']['freq'])
            #test_metrics = model.test(df)
        
            return {'loss':train_metrics['RMSE'].reset_index(drop=True)[0], 'status': STATUS_OK }
        except Exception as e:
            print(f"Exception in objective function: {e}")
            return {'loss': 1000, 'status': 'fail'}

    early_stop_fn = no_progress_loss(iteration_stop_count=int((args['ip_params']['max_evals'])*0.2), percent_increase=5)
    trials = Trials()
    best_results = fmin(optimize, space=args, algo=tpe.suggest, trials=trials, max_evals=args['ip_params']['max_evals'], early_stop_fn = early_stop_fn, show_progressbar=False)

    return best_results

def select_yhat(row, yhat_columns):
    for col in yhat_columns:
        if pd.notna(row[col]) and row[col] != 0:
            return row[col]
    return np.nan

# def train_neural_prophet(df, df_test, model_params, ip_params, op_params, test_horizon):
        
#     # Combine model parameters & additional input & output parameters
#     args = {'model_params':model_params,'ip_params':ip_params,'op_params':op_params} 
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # GPU Check        
#     print('Device Used: {}'.format(device))
#     trainer_config = {"accelerator":"cuda"} # use GPU if available, no need for model.to(device) for neuralprophet           
    
#     def optimize(args): # Hyperparameter tuning with Hyperopt
#         df = args['ip_params']['df']
#         if 'ID' in df.columns: # add local trend & seasonality for each ID if present
#             global_local = 'local'
#         else: # else model the whole series as a whole
#             global_local = 'global'            
#         model = NeuralProphet( **{**args['model_params'],**args['op_params']},trend_global_local=global_local,\
#                                                                             season_global_local=global_local)        
#         if args['ip_params']['lagged_regressor_cols'] is not None:
#             if args['op_params']['n_lags']>0:
#                 for col in args['ip_params']['lagged_regressor_cols']:
#                     model = model.add_lagged_regressor(col, normalize="standardize")
#             else:
#                 df = df[list(set(df.columns) - set(ip_params['lagged_regressor_cols']))]
#         df_train, df_val = model.split_df(df, freq=args['ip_params']['freq'], valid_p=args['ip_params']['valid_p'])
#         train_metrics = model.fit(df_train, freq=args['ip_params']['freq'], validation_df=df_val)
#         test_metrics = model.test(df_val)
    
#         return {'loss':test_metrics['RMSE'].reset_index(drop=True)[0], 'status': STATUS_OK }

#     early_stop_fn = no_progress_loss(iteration_stop_count=int((args['ip_params']['max_evals'])*0.7), percent_increase=5)
#     trials = Trials()
#     best_results = fmin(optimize, space=args, algo=tpe.suggest, trials=trials, max_evals=args['ip_params']['max_evals'],\
#                        early_stop_fn = early_stop_fn)
                       
#     best_model_params =\
#     {
#     # 'epochs':epochs[best_results['epochs']], 
#     # 'daily_seasonality':daily_seasonality[best_results['daily_seasonality']],
#     # 'weekly_seasonality':weekly_seasonality[best_results['weekly_seasonality']],
#     'yearly_seasonality':best_results['yearly_seasonality'],
#     # 'loss_func':loss_func[best_results['loss_func']],
#     'seasonality_mode':best_results['seasonality_mode'], 
#     'n_changepoints':best_results['n_changepoints'],
#     # 'learning_rate':learning_rate[best_results['learning_rate']], 
#     }
    
#     df = args['ip_params']['df']    
#     # if 'ID' in df.columns: # add local trend & seasonality for each ID if present
#     #     global_local = 'local'
#     # else: # else model the whole series as a whole
#     #     global_local = 'global'                
#     # model = NeuralProphet( **{**best_model_params,**args['op_params']},trend_global_local=global_local,\
#     #                                                                           season_global_local=global_local) 
    
#     if args['ip_params']['lagged_regressor_cols'] is not None:    
#         if args['op_params']['n_lags']>0:
#             for col in args['ip_params']['lagged_regressor_cols']:
#                 model = model.add_lagged_regressor(col, normalize="standardize")
#         else:
#             df = df[list(set(df.columns) - set(ip_params['lagged_regressor_cols']))]        
    
#     train_test_splits = rolling_origin_func(pd.concat([df, df_test], axis=0).reset_index(drop=True), test_horizon, roll_step=1)

#     rmse_n, mape_n = [], []
#     n_training_days = []

#     # loop through train/test splits
#     for x in train_test_splits:
#         train, test = x
#         n_training_days.append(len(train.index))

#         print(train)
#         print(test)

#         # train NeuralProphet and get accuracy 
#         model = NeuralProphet(n_lags = args['op_params']['n_lags'], n_forecasts = test_horizon, **best_model_params)

#         model.fit(train, freq=args['ip_params']['freq'])
#         future = model.make_future_dataframe(train, periods=args['ip_params']['periods'])
#         forecast = model.predict(future, raw=True, decompose=False)
#         rmse, mape = accuracy(test['y'], pd.Series(np.array(forecast.iloc[:, 1:]).flatten()))
#         rmse_n.append(rmse)
#         mape_n.append(mape)
    
#     # train_metrics = model.fit(df, freq=args['ip_params']['freq'])
#     # df_test = pd.concat([df.iloc[-args['op_params']['n_lags']:], df_test], ignore_index=True)
#     # test_metrics = model.test(df_test)    
#     # future = model.make_future_dataframe(df, periods=args['ip_params']['periods'],\
#     #                                          n_historic_predictions=args['ip_params']['n_historic_predictions'])
#     # forecast = model.predict(future)
#     # final_train_metrics = train_metrics.iloc[-1:].reset_index(drop=True)
#     # final_test_metrics = test_metrics.iloc[-1:].reset_index(drop=True)
#     return best_model_params, rmse_n, mape_n

# def select_yhat(row, yhat_columns):
#     for col in yhat_columns:
#         if pd.notna(row[col]) and row[col] != 0:
#             return row[col]
#     return np.nan
