from statsmodels.tsa.stattools import adfuller

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

def find_optimal_lags(series, alpha):
    autocorr, confint = acf(series.dropna(), alpha=alpha, nlags=100, fft=True)
    conf_offset = confint[:, 1] - autocorr
    optimal_lags = np.where((autocorr < conf_offset) & (autocorr > -conf_offset))[0]

    if len(optimal_lags) == 0:
        return 0
    else:
        return optimal_lags[0] - 1

def calculate_optimal_lags(df, alpha=0.05):
    optimal_lags_dict = {}
    if 'ID' in df.columns:
        for id_value in df['ID'].unique():
            series = df[df['ID'] == id_value]['y']
            optimal_lags_dict[id_value] = find_optimal_lags(series, alpha)
    else:
        series = df['y']
        optimal_lags_dict['1'] = find_optimal_lags(series, alpha)

    return optimal_lags_dict

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

def train_neural_prophet(df, model_params, ip_params, op_params):    
    # Combine model parameters & additional input & output parameters
    args = {'model_params':model_params,'ip_params':ip_params,'op_params':op_params} 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # GPU Check        
    print('Device Used: {}'.format(device))
    trainer_config = {"accelerator":"cuda"} # use GPU if available, no need for model.to(device) for neuralprophet           
    
    def optimize(args): # Hyperparameter tuning with Hyperopt
        df = args['ip_params']['df']
        if 'ID' in df.columns: # add local trend & seasonality for each ID if present
            global_local = 'local'
        else: # else model the whole series as a whole
            global_local = 'global'            
        model = NeuralProphet( **{**args['model_params'],**args['op_params']},trend_global_local=global_local,\
                                                                              season_global_local=global_local)        
        if args['ip_params']['lagged_regressor_cols'] is not None:
            if args['op_params']['n_lags']>0:
                for col in args['ip_params']['lagged_regressor_cols']:
                    model = model.add_lagged_regressor(col, normalize="standardize")
            else:
                df = df[list(set(df.columns) - set(ip_params['lagged_regressor_cols']))]
        df_train, df_test = model.split_df(df, freq=args['ip_params']['freq'], valid_p=args['ip_params']['valid_p'])
        train_metrics = model.fit(df_train, freq=args['ip_params']['freq'])
        test_metrics = model.test(df_test)    
        return {'loss':test_metrics['RMSE_val'].reset_index(drop=True)[0], 'status': STATUS_OK }

    early_stop_fn = no_progress_loss(iteration_stop_count=int((args['ip_params']['max_evals'])*0.7), percent_increase=0.5)
    trials = Trials()
    best_results = fmin(optimize, space=args, algo=tpe.suggest, trials=trials, max_evals=args['ip_params']['max_evals'],\
                       early_stop_fn = early_stop_fn)
    best_model_params =\
    {
    'epochs':epochs[best_results['epochs']], 
    'daily_seasonality':daily_seasonality[best_results['daily_seasonality']],
    'weekly_seasonality':weekly_seasonality[best_results['weekly_seasonality']],
    'yearly_seasonality':yearly_seasonality[best_results['yearly_seasonality']],
    'loss_func':loss_func[best_results['loss_func']],
    'seasonality_mode':seasonality_mode[best_results['seasonality_mode']], 
    'n_changepoints':n_changepoints[best_results['n_changepoints']],
    'learning_rate':learning_rate[best_results['learning_rate']], 
    }
    
    df = args['ip_params']['df']    
    if 'ID' in df.columns: # add local trend & seasonality for each ID if present
        global_local = 'local'
    else: # else model the whole series as a whole
        global_local = 'global'                
    model = NeuralProphet( **{**best_model_params,**args['op_params']},trend_global_local=global_local,\
                                                                              season_global_local=global_local) 
        
    if args['ip_params']['lagged_regressor_cols'] is not None:    
        if args['op_params']['n_lags']>0:
            for col in args['ip_params']['lagged_regressor_cols']:
                model = model.add_lagged_regressor(col, normalize="standardize")
        else:
            df = df[list(set(df.columns) - set(ip_params['lagged_regressor_cols']))]        
    df_train, df_test = model.split_df(df, freq=args['ip_params']['freq'], valid_p=args['ip_params']['valid_p'])
    train_metrics = model.fit(df_train, freq=args['ip_params']['freq'])
    test_metrics = model.test(df_test)    
    future = model.make_future_dataframe(df, periods=args['ip_params']['periods'],\
                                             n_historic_predictions=args['ip_params']['n_historic_predictions'])
    forecast = model.predict(future)
    final_train_metrics = train_metrics.iloc[-1:].reset_index(drop=True)
    final_test_metrics = test_metrics.iloc[-1:].reset_index(drop=True)
    return model, forecast, best_model_params, final_train_metrics, final_test_metrics

def select_yhat(row, yhat_columns):
    for col in yhat_columns:
        if pd.notna(row[col]) and row[col] != 0:
            return row[col]
    return np.nan
