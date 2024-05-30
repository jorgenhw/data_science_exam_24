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
```
.
├── .gitignore
├── ARIMA
│   ├── arima_test.py
│   ├── arima_tune.py
│   ├── outputs
│   │   ├── *All results predictions, error metrics and parameters from HP tuning*
│   └── utils.py
├── LagLlama
│   ├── lag_llama_test.py
│   ├── lag_llama_tune.py
│   ├── lagllama_setup.sh
│   ├── outputs
│   │   ├── *All results predictions, error metrics and HP parameters*
│   └── utils
│       └── lag_utils.py
├── NeuralProphet
│   ├── neuralprophet_test.py
│   ├── neuralprophet_tune.py
│   ├── outputs
│   │   ├── *All results predictions, error metrics and parameters from HP tuning*
│   ├── readme.md
│   └── utils.py
├── README.md
├── RNN
│   ├── analysis_climate.ipynb
│   ├── analysis_weather.ipynb
│   ├── error_metrics.ipynb
│   └── results
│       ├── all_error_metrics
│       │   └── all_error_metrics.csv
│       ├── climate_data
│       │   ├── error_metrics
│       │   │   └── *All error metrics in .csv*
│       │   └── predictions
│       │       └── *All predictions in .csv*
│       └── weather_data
│       │   ├── error_metrics
│       │   │   └── *All error metrics in .csv*
│       │   └── predictions
│       │       └── *All predictions in .csv*
├── TimeGPT
│   ├── analysis_climate.ipynb
│   ├── analysis_weather.ipynb
│   ├── error_metrics.ipynb
│   └── results
│       ├── all_error_metrics
│       │   └── all_error_metrics.csv
│       ├── climate_data
│       │   ├── error_metrics
│       │   │   └── *All error metrics in .csv*
│       │   └── predictions
│       │       └── *All predictions in .csv*
│       └── weather_data
│       │   ├── error_metrics
│       │   │   └── *All error metrics in .csv*
│       │   └── predictions
│       │       └── *All predictions in .csv*
├── data
│   ├── climate
│   │   ├── AMOCdata.csv
│   │   ├── data_partitioning.ipynb
│   │   ├── readme.md
│   │   └── splits
│   │       ├── *contains test and train splits of data*
│   ├── data_visualisation.ipynb
│   ├── figures
│   │   ├── climate.png
│   │   └── weather.png
│   ├── splitting_datasets.ipynb
│   └── weather
│       ├── data_partitioning.ipynb
│       ├── splits
│       │   ├── *contains test and train splits of data*
├── requirements.txt
├── setup.sh
└── src
    ├── forecast_comparison.png
    ├── result_plots.ipynb
    ├── rolling_origin_func.py
    └── visualizations.ipynb
```