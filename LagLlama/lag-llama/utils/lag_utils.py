import torch
from lag_llama.gluon.estimator import LagLlamaEstimator
from gluonts.evaluation import make_evaluation_predictions
import pandas as pd
from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName
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

def prepare_data_for_lag_llama(train, test):
    df_train = pd.read_csv(f'data_test/climate/splits/train/train_{train}.csv')
    df_test = pd.read_csv(f'data_test/climate/splits/test/test_{test}.csv')

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

    df_train.set_index('ds', inplace=True)
    df_test.set_index('ds', inplace=True)
    
    return df_train, df_test

def to_deepar_format(dataframe, freq):
    start_index = dataframe.index.min()
    data = [{
                FieldName.START:  start_index,
                FieldName.TARGET:  dataframe[c].values,
            }
            for c in dataframe.columns]
    #print(data[0])
    return ListDataset(data, freq=freq)

def get_lag_llama_predictions(dataset, prediction_length, device, context_length=32, use_rope_scaling=False, num_samples=100):
    ckpt = torch.load("lag-llama/lag-llama.ckpt", map_location=device) # Uses GPU since in this Colab we use a GPU.
    estimator_args = ckpt["hyper_parameters"]["model_kwargs"]

    rope_scaling_arguments = {
        "type": "linear",
        "factor": max(1.0, (context_length + prediction_length) / estimator_args["context_length"]),
    }

    estimator = LagLlamaEstimator(
        ckpt_path="lag-llama/lag-llama.ckpt",
        prediction_length=prediction_length,
        context_length=context_length, # Lag-Llama was trained with a context length of 32, but can work with any context length

        # estimator args
        input_size=estimator_args["input_size"],
        n_layer=estimator_args["n_layer"],
        n_embd_per_head=estimator_args["n_embd_per_head"],
        n_head=estimator_args["n_head"],
        scaling=estimator_args["scaling"],
        time_feat=estimator_args["time_feat"],
        rope_scaling=rope_scaling_arguments if use_rope_scaling else None,

        batch_size=1,
        num_parallel_samples=100,
        device=device,
    )

    lightning_module = estimator.create_lightning_module()
    transformation = estimator.create_transformation()
    predictor = estimator.create_predictor(transformation, lightning_module)

    forecast_it, ts_it = make_evaluation_predictions(
        dataset=dataset,
        predictor=predictor,
        num_samples=num_samples
    )
    forecasts = list(forecast_it)
    tss = list(ts_it)



    return forecasts, tss