import pmdarima as pm
from pmdarima import model_selection
import pandas as pd
from sklearn.metrics import mean_squared_error
from mango import Tuner
import numpy as np
import sys

train_size = [50, 250, 750]
test_size = [12, 120, 360]

train = train_size[int(sys.argv[1])]
test = test_size[int(sys.argv[2])]

df_train = pd.read_csv(f'data/climate/data_partitions/Partition_test_{test}_train_{train}_train.csv')
df_test = pd.read_csv(f'data/climate/data_partitions/Partition_test_{test}_train_{train}_test.csv')

def arima_objective_function(args_list):
    global df_train
    
    val_num = int(len(df_train)*0.2)

    cv = model_selection.RollingForecastCV(h=val_num, step=val_num, initial=val_num)

    params_evaluated = []
    model_cv_score_list = []
    
    for params in args_list:
        try:
            p,d,q = params['p'],params['d'], params['q']
            trend = params['trend']

            model = pm.ARIMA(order=(p,d,q), trend = trend)
            
            model_cv_scores = model_selection.cross_val_score(model, df_train['AMOC0'], scoring='mean_squared_error', cv=cv, verbose=2)
            model_cv_score_list.append(np.mean(model_cv_scores))  
            params_evaluated.append(params)

        except:
            #print(f"Exception raised for {params}")
            #pass 
            params_evaluated.append(params)
            model_cv_score_list.append(1e5)
        
    return params_evaluated, model_cv_score_list

param_space = dict(p= range(0, 50),
                   d= range(0, 50),
                   q =range(0, 50),
                   trend = ['n', 'c', 't', 'ct']
                  )

conf_Dict = dict()
conf_Dict['num_iteration'] = 200
tuner = Tuner(param_space, arima_objective_function, conf_Dict)
results = tuner.minimize()
print('best parameters:', results['best_params'])
print('best loss:', results['best_objective'])

order = (results['best_params']['p'], results['best_params']['d'], results['best_params']['q'])
trend = results['best_params']['trend']

model = pm.ARIMA(order=order, trend=trend)
model.fit(df_train['AMOC0'])
prediction = model.predict(n_periods=len(df_test['AMOC0']))

# save data and predictions in pandas dataframe
if len(df_test) < 50:
    forecast_horizon = 'short'
elif len(df_test) < 300:
    forecast_horizon = 'medium'
else:
    forecast_horizon = 'long'

# select only date, amoc0, and prediction columns
df_test = df_test[['date', 'AMOC0']]
df_test['prediction'] = prediction
df_test.to_csv(f'ARIMA/data_sea/{len(df_train)}_rows/{forecast_horizon}/Partition_test_{len(df_test)}_train_{len(df_train)}_errors.csv', index=False)
