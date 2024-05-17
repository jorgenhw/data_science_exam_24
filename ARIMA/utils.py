
import pandas as pd
from typing import List, Tuple
from sklearn.metrics import mean_squared_error
from math import sqrt

# def rmse_criterion(y, y_pred):
#     return -sqrt(mean_squared_error(y, y_pred))

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

def prepare_data_for_arima(train, test):
    df_train = pd.read_csv(f'data/climate/splits/train/train_{train}.csv')
    df_test = pd.read_csv(f'data/climate/splits/test/test_{test}.csv')

    # rename columns to fit neural prophet requirements
    df_train.rename(columns={'date': 'ds', 'AMOC0': 'y'}, inplace=True)
    df_test.rename(columns={'date': 'ds', 'AMOC0': 'y'}, inplace=True)

    # remove columns that are not needed
    df_train.drop(columns=['time', 'AMOC1', 'AMOC2', 'GM'], inplace=True)
    df_test.drop(columns=['time', 'AMOC1', 'AMOC2', 'GM'], inplace=True)

    df_train['ds'] = pd.to_datetime(df_train['ds'])
    df_test['ds'] = pd.to_datetime(df_test['ds'])

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

    # df_train.set_index('ds', inplace=True)
    # df_test.set_index('ds', inplace=True)
    
    return df_train, df_test