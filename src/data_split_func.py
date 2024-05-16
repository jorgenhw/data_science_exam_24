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