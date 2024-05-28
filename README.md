<!-- PROJECT LOGO -->
<br />
<p align="center">
  <h1 align="center">A Time-Series Showdown</h1> 
  <h2 align="center">An evaluation and comparison of different methodologies within time-series analysis</h2> 
  <h3 align="center">Cognitive Science, Aarhus University</h3> 
  <h3 align="center">Data Science Exam</h3> 
  <p align="center">
    Jørgen Højlund Wibe (201807750)<br>
    Niels Aalund Krogsgaard ()
  </p>
</p>


<!-- ABOUT THE PROJECT -->
## About the project
Bla bla bla


<!-- USAGE -->
## Usage
To use or reproduce the results you need to adopt the following steps.

**NOTE:** There may be slight variations depending on the terminal and operating system you use. The following example is designed to work using the Visual Studio Code version 1.77.3 (Universal). The terminal code should therefore work using a unix-based bash. The avoid potential package conflicts, the ```setup.sh``` bash files contains the steps necesarry to create a virtual environment, install libraries and run the project.


1. Do x
2. then do y
3. Run ```setup.sh```

## Repository structure
.
├── .gitignore
├── ARIMA
│   ├── arima_test.py
│   ├── arima_tune.py
│   ├── outputs
│   │   ├── test
│   │   │   ├── Partition_horizon_10_train_large_climate
│   │   │   │   └── forecast_df.csv
│   │   │   ├── Partition_horizon_10_train_large_weather
│   │   │   │   └── forecast_df.csv
│   │   │   ├── Partition_horizon_10_train_small_climate
│   │   │   │   └── forecast_df.csv
│   │   │   ├── Partition_horizon_10_train_small_weather
│   │   │   │   └── forecast_df.csv
│   │   │   ├── Partition_horizon_50_train_large_climate
│   │   │   │   └── forecast_df.csv
│   │   │   ├── Partition_horizon_50_train_large_weather
│   │   │   │   └── forecast_df.csv
│   │   │   ├── Partition_horizon_50_train_small_climate
│   │   │   │   └── forecast_df.csv
│   │   │   ├── Partition_horizon_50_train_small_weather
│   │   │   │   └── forecast_df.csv
│   │   │   ├── arima_results.csv
│   │   │   └── arima_results_summary.csv
│   │   └── tuning
│   │       └── arima_hyperparameters.csv
│   └── utils.py
├── LagLlama
│   ├── lag_llama_test.py
│   ├── lag_llama_tune.py
│   ├── lagllama_setup.sh
│   ├── outputs
│   │   ├── test
│   │   │   ├── Partition_horizon_10_train_large_climate
│   │   │   │   └── forecast_df.csv
│   │   │   ├── Partition_horizon_10_train_large_weather
│   │   │   │   └── forecast_df.csv
│   │   │   ├── Partition_horizon_10_train_small_climate
│   │   │   │   └── forecast_df.csv
│   │   │   ├── Partition_horizon_10_train_small_weather
│   │   │   │   └── forecast_df.csv
│   │   │   ├── Partition_horizon_50_train_large_climate
│   │   │   │   └── forecast_df.csv
│   │   │   ├── Partition_horizon_50_train_large_weather
│   │   │   │   └── forecast_df.csv
│   │   │   ├── Partition_horizon_50_train_small_climate
│   │   │   │   └── forecast_df.csv
│   │   │   ├── Partition_horizon_50_train_small_weather
│   │   │   │   └── forecast_df.csv
│   │   │   ├── all_metrics.csv
│   │   │   ├── all_rmse.csv
│   │   │   └── temp.py
│   │   └── tuning
│   │       ├── all_metrics_climate_large_10_128_[True, False].csv
│   │       ├── all_metrics_climate_large_10_256_[True, False].csv
│   │       ├── all_metrics_climate_large_10_32_[True, False].csv
│   │       ├── all_metrics_climate_large_10_512_[True, False].csv
│   │       ├── all_metrics_climate_large_10_64_[True, False].csv
│   │       ├── all_metrics_climate_large_10_950_[True, False].csv
│   │       ├── all_metrics_climate_large_50_128_[True, False].csv
│   │       ├── all_metrics_climate_large_50_256_[True, False].csv
│   │       ├── all_metrics_climate_large_50_32_[True, False].csv
│   │       ├── all_metrics_climate_large_50_512_[True, False].csv
│   │       ├── all_metrics_climate_large_50_64_[True, False].csv
│   │       ├── all_metrics_climate_large_50_950_[True, False].csv
│   │       ├── all_metrics_weather_large_10_128_[True, False].csv
│   │       ├── all_metrics_weather_large_10_256_[True, False].csv
│   │       ├── all_metrics_weather_large_10_32_[True, False].csv
│   │       ├── all_metrics_weather_large_10_512_[True, False].csv
│   │       ├── all_metrics_weather_large_10_64_[True, False].csv
│   │       ├── all_metrics_weather_large_10_950_[True, False].csv
│   │       ├── all_metrics_weather_large_50_128_[True, False].csv
│   │       ├── all_metrics_weather_large_50_256_[True, False].csv
│   │       ├── all_metrics_weather_large_50_32_[True, False].csv
│   │       ├── all_metrics_weather_large_50_512_[True, False].csv
│   │       ├── all_metrics_weather_large_50_64_[True, False].csv
│   │       ├── all_metrics_weather_large_50_950_[True, False].csv
│   │       └── best_hyperparameters.csv
│   └── utils
│       └── lag_utils.py
├── NeuralProphet
│   ├── neuralprophet_test.py
│   ├── neuralprophet_tune.py
│   ├── outputs
│   │   ├── test
│   │   │   ├── Partition_horizon_10_train_large_climate
│   │   │   │   └── forecast_df.csv
│   │   │   ├── Partition_horizon_10_train_large_weather
│   │   │   │   └── forecast_df.csv
│   │   │   ├── Partition_horizon_10_train_small_climate
│   │   │   │   └── forecast_df.csv
│   │   │   ├── Partition_horizon_10_train_small_weather
│   │   │   │   └── forecast_df.csv
│   │   │   ├── Partition_horizon_50_train_large_climate
│   │   │   │   └── forecast_df.csv
│   │   │   ├── Partition_horizon_50_train_large_weather
│   │   │   │   └── forecast_df.csv
│   │   │   ├── Partition_horizon_50_train_small_climate
│   │   │   │   └── forecast_df.csv
│   │   │   ├── Partition_horizon_50_train_small_weather
│   │   │   │   └── forecast_df.csv
│   │   │   ├── all_metrics.csv
│   │   │   └── np_summary.csv
│   │   └── tuning
│   │       ├── all_metrics_climate_large_10.csv
│   │       ├── all_metrics_climate_large_50.csv
│   │       ├── all_metrics_climate_small_10.csv
│   │       ├── all_metrics_climate_small_50.csv
│   │       ├── all_metrics_weather_large_10.csv
│   │       ├── all_metrics_weather_large_50.csv
│   │       ├── all_metrics_weather_small_10.csv
│   │       ├── all_metrics_weather_small_50.csv
│   │       ├── best_hyperparameters.csv
│   │       ├── my_variable_0.pkl
│   │       ├── my_variable_1.pkl
│   │       ├── my_variable_2.pkl
│   │       ├── my_variable_3.pkl
│   │       ├── my_variable_4.pkl
│   │       ├── my_variable_5.pkl
│   │       ├── my_variable_6.pkl
│   │       └── my_variable_7.pkl
│   ├── readme.md
│   └── utils.py
├── README.md
├── RNN
│   ├── analysis_climate.ipynb
│   ├── analysis_weather.ipynb
│   ├── error_metrics
│   │   ├── all_error_metrics.csv
│   │   ├── train_1000_test_250_horizon_50_results_climate.csv
│   │   ├── train_1000_test_250_horizon_50_results_weather.csv
│   │   ├── train_1000_test_50_horizon_10_results_climate.csv
│   │   ├── train_1000_test_50_horizon_10_results_weather.csv
│   │   ├── train_100_test_250_horizon_50_results_climate.csv
│   │   ├── train_100_test_250_horizon_50_results_weather.csv
│   │   ├── train_100_test_50_horizon_10_results_climate.csv
│   │   └── train_100_test_50_horizon_10_results_weather.csv
│   ├── error_metrics.ipynb
│   └── results
│       ├── climate_data
│       │   ├── train_1000_test_250_horizon_50_results.csv
│       │   ├── train_1000_test_50_horizon_10_results.csv
│       │   ├── train_100_test_250_horizon_50_results.csv
│       │   └── train_100_test_50_horizon_10_results.csv
│       └── weather_data
│           ├── train_1000_test_250_horizon_50_results.csv
│           ├── train_1000_test_50_horizon_10_results.csv
│           ├── train_100_test_250_horizon_50_results.csv
│           └── train_100_test_50_horizon_10_results.csv
├── TimeGPT
│   ├── analysis_climate.ipynb
│   ├── analysis_climate_w_finetune.ipynb
│   ├── analysis_weather.ipynb
│   ├── analysis_weather_w_finetune.ipynb
│   ├── error_metrics
│   │   ├── combined_error_results.csv
│   │   ├── train_1000_test_250_horizon_50_modeltype_timegpt-1-long-horizon_climate_results.csv
│   │   ├── train_1000_test_250_horizon_50_weather_results_weather.csv
│   │   ├── train_1000_test_50_horizon_10_modeltype_timegpt-1_climate_results.csv
│   │   ├── train_1000_test_50_horizon_10_modeltype_timegpt-1_climate_resultsTESTTTT.csv
│   │   ├── train_1000_test_50_horizon_10_weather_results_weather.csv
│   │   ├── train_1008_test_250_horizon_50_weather__WITH_FINETUNING_results_weather.csv
│   │   ├── train_100_test_250_horizon_50_modeltype_timegpt-1-long-horizon_climate_results.csv
│   │   ├── train_100_test_250_horizon_50_weather_results_weather.csv
│   │   ├── train_100_test_50_horizon_10_model_timegpt-1_weather_results_weather.csv
│   │   ├── train_100_test_50_horizon_10_modeltype_timegpt-1_climate_results.csv
│   │   └── weather_datatrain_1008_test_250_horizon_50_weather__WITH_FINETUNING_results_weather.csv
│   ├── error_metrics.ipynb
│   └── results
│       ├── climate_data
│       │   ├── train_1000_test_250_horizon_50_modeltype_timegpt-1-long-horizon_climate_results.csv
│       │   ├── train_1000_test_50_horizon_10_modeltype_timegpt-1_climate_results.csv
│       │   ├── train_100_test_250_horizon_50_modeltype_timegpt-1-long-horizon_climate_results.csv
│       │   └── train_100_test_50_horizon_10_modeltype_timegpt-1_climate_results.csv
│       ├── combined_csv.csv
│       ├── train_1008_test_250_horizon_50_weather__WITH_FINETUNING_results.csv
│       ├── weather_data
│       │   ├── train_1000_test_250_horizon_50_weather_results.csv
│       │   ├── train_1000_test_50_horizon_10_weather_results.csv
│       │   ├── train_100_test_250_horizon_50_weather_results.csv
│       │   └── train_100_test_50_horizon_10_model_timegpt-1_weather_results.csv
│       └── weather_datatrain_1008_test_250_horizon_50_weather__WITH_FINETUNING_results.csv
├── data
│   ├── climate
│   │   ├── AMOCdata.csv
│   │   ├── data_partitioning.ipynb
│   │   ├── readme.md
│   │   └── splits
│   │       ├── test
│   │       │   ├── test_large_for_train_large.csv
│   │       │   ├── test_large_for_train_small.csv
│   │       │   ├── test_small_for_train_large.csv
│   │       │   └── test_small_for_train_small.csv
│   │       └── train
│   │           ├── train_large.csv
│   │           └── train_small.csv
│   ├── data_visualisation.ipynb
│   ├── figures
│   │   ├── climate.png
│   │   └── weather.png
│   ├── splitting_datasets.ipynb
│   └── weather
│       ├── data_partitioning.ipynb
│       ├── splits
│       │   ├── test
│       │   │   ├── test_large_for_train_large.csv
│       │   │   ├── test_large_for_train_small.csv
│       │   │   ├── test_small_for_train_large.csv
│       │   │   └── test_small_for_train_small.csv
│       │   └── train
│       │       ├── train_large.csv
│       │       └── train_small.csv
│       └── weatherAarhusSyd_Marts_april.csv
├── requirements.txt
├── setup.sh
└── src
    ├── forecast_comparison.png
    ├── result_plots.ipynb
    └── rolling_origin_func.py