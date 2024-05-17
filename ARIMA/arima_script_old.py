import pmdarima as pm
from pmdarima import model_selection
import pandas as pd
from sklearn.metrics import mean_squared_error
#from mango import Tuner
import numpy as np
import sys
from utils import *
from pmdarima import auto_arima

train_size = ['small', 'large']
test_size = ['small', 'large']
forecast_horizons = [10, 50]

# train_size = ['small']
# test_size = ['small']
# forecast_horizons = [10]

all_metrics = []

i = 1


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

        df_train, df_test = prepare_data_for_arima(train, test)

        # Search for optimal ARIMA parameters
        model = auto_arima(df_train['y'], start_p=1, start_q=1,
                        max_p=50, max_q=50, m=12,
                        start_P=0, seasonal=True,
                        d=None, D=None, trace=True,
                        error_action='ignore',  
                        suppress_warnings=True, 
                        stepwise=True, method='nm',
                        maxiter=30)
        
        model_params = model.get_params()

        # remove maxiter, method, out_of_sample_size, scoring, scoring_args, start_params, and supress_warnings from model_params
        model_params.pop('maxiter')
        model_params.pop('method')
        model_params.pop('out_of_sample_size')
        model_params.pop('scoring')
        model_params.pop('scoring_args')
        model_params.pop('start_params')
        model_params.pop('suppress_warnings')

        # add information about hyperparameters to all_metrics
        hyperparameters_dict = {
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

# save all_metrics_df to csv
all_metrics_df.to_csv('ARIMA/output/tuning/arima_hyperparameters.csv', index=False)


        


        # def arima_objective_function(args_list):
        #     global df_train
            
        #     val_num = int(len(df_train)*0.2)

        #     cv = model_selection.RollingForecastCV(h=val_num, step=val_num, initial=val_num)

        #     params_evaluated = []
        #     model_cv_score_list = []
            
        #     for params in args_list:
        #         try:
        #             p,d,q = params['p'],params['d'], params['q']
        #             trend = params['trend']

        #             model = pm.ARIMA(order=(p,d,q), trend = trend)
                    
        #             model_cv_scores = model_selection.cross_val_score(model, df_train['AMOC0'], scoring='mean_squared_error', cv=cv, verbose=2)
        #             model_cv_score_list.append(np.mean(model_cv_scores))  
        #             params_evaluated.append(params)

        #         except:
        #             #print(f"Exception raised for {params}")
        #             #pass 
        #             params_evaluated.append(params)
        #             model_cv_score_list.append(1e5)
                
        #     return params_evaluated, model_cv_score_list

        # param_space = dict(p= range(0, 50),
        #                 d= range(0, 50),
        #                 q =range(0, 50),
        #                 trend = ['n', 'c', 't', 'ct']
        #                 )

        # conf_Dict = dict()
        # conf_Dict['num_iteration'] = 200
        # tuner = Tuner(param_space, arima_objective_function, conf_Dict)
        # results = tuner.minimize()
        # print('best parameters:', results['best_params'])
        # print('best loss:', results['best_objective'])

        # order = (results['best_params']['p'], results['best_params']['d'], results['best_params']['q'])
        # trend = results['best_params']['trend']

# model = pm.ARIMA(order=order, trend=trend)
# model.fit(df_train['AMOC0'])
# prediction = model.predict(n_periods=len(df_test['AMOC0']))

# # save data and predictions in pandas dataframe
# if len(df_test) < 50:
#     forecast_horizon = 'short'
# elif len(df_test) < 300:
#     forecast_horizon = 'medium'
# else:
#     forecast_horizon = 'long'

# # select only date, amoc0, and prediction columns
# df_test = df_test[['date', 'AMOC0']]
# df_test['prediction'] = prediction
# df_test.to_csv(f'ARIMA/data_sea/{len(df_train)}_rows/{forecast_horizon}/Partition_test_{len(df_test)}_train_{len(df_train)}_errors.csv', index=False)
