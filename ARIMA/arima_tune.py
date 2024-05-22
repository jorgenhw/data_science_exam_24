from pmdarima import auto_arima
from utils import *
import os

dataset = ['climate','weather']
train_size = ['small', 'large']
test_size = ['small', 'large']
forecast_horizons = [10, 50]

all_metrics = []

for data in dataset:
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

            df_train, df_test = prepare_data_for_arima(data, train, test)

            if data == 'climate':
                df_train, df_test = fix_dates_month(df_train, df_test)
                m = 12
            elif data == 'weather':
                m = 24

            # Search for optimal ARIMA parameters
            model = auto_arima(df_train['y'], start_p=1, start_q=1,
                            max_p=20, max_q=20, m=m,
                            start_P=0, seasonal=True,
                            d=None, D=None, trace=True,
                            error_action='ignore',  
                            suppress_warnings=True, 
                            stepwise=False, method='lbfgs',
                            maxiter=50, out_of_sample_size=forecast_horizon,
                            information_criterion='oob', scoring='mse')
            
            model_params = model.get_params()

            # remove maxiter, method, out_of_sample_size, scoring, scoring_args, start_params, and supress_warnings from model_params
            model_params.pop('maxiter')
            model_params.pop('method')
            model_params.pop('out_of_sample_size')
            model_params.pop('scoring')
            model_params.pop('scoring_args')
            model_params.pop('start_params')
            model_params.pop('suppress_warnings')

            model_dict = model.to_dict()
            # choose only oob from model_dict and save it to model_params
            model_params['mse'] = model_dict['oob']

            # add information about hyperparameters to all_metrics
            hyperparameters_dict = {
                'dataset': data,
                'train_size': train,
                'test_size': test,
                'forecast_horizon': forecast_horizon
            }

            # combine hyperparameters with metrics
            model_params.update(hyperparameters_dict)

            # add dict to all_metrics
            all_metrics.append(model_params)

# make all_metrics into a pandas dataframe
all_metrics_df = pd.DataFrame(all_metrics)

# create directory and save all outputs
os.makedirs(f'ARIMA/outputs/tuning', exist_ok=True)

# save all_metrics_df to csv
all_metrics_df.to_csv('ARIMA/outputs/tuning/arima_hyperparameters.csv', index=False)
