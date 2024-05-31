<!-- PROJECT LOGO -->
<br />
<p align="center">
  <h1 align="center">A Time-Series Showdown</h1> 
  <h2 align="center">An evaluation and comparison of different methodologies within time-series analysis</h2> 
  <h3 align="center">Cognitive Science, Aarhus University</h3> 
  <h3 align="center">Data Science Exam</h3> 
  <p align="center">
    Jørgen Højlund Wibe (201807750)<br>
    Niels Aalund Krogsgaard (202008114)
  </p>
</p>


<!-- ABOUT THE PROJECT -->
## About the project
This repository contains all the data and code necesarry to reproduce the results from a paper comparing different time-series methodolgies. In more detail, we investigate traditional time-series forecasting models, such as ARIMA, LSTM, and NeuralProphet, against newer foundation models like TimeGPT and LagLlama. Using two diverse datasets — weather patterns from Aarhus and Atlantic sea surface temperatures — we assess model performance across different training data sizes and forecast horizons. Our findings reveal that while no single model is universally superior, foundation models like TimeGPT show significant promise, often matching or outperforming traditional methods in accuracy and ease of use. However, the study also identifies several challenges on issues related to benchmarking standardization. These insights emphasize the need for robust evaluation frameworks and further research into optimizing foundation models for specific applications. Ultimately, this research highlights the evolving landscape of time-series forecasting and its implications for effective decision-making and resource allocation.


<!-- USAGE -->
## Usage
To use or reproduce the results you need to adopt the following steps.

**NOTE:** There may be slight variations depending on the terminal and operating system you use. The following example is designed to work using the Visual Studio Code version 1.77.3 (Universal). The terminal code should therefore work using a unix-based bash. The avoid potential package conflicts, the ```setup.sh``` bash files contains the steps necesarry to create a virtual environment, install libraries and run the project.

1. Clone repository
2. Run setup.sh
3. Open the notebook of your choice

> **Step 1** Clone repository

Clone repository using the following lines in the unix-based bash:

```bash
git clone https://github.com/jorgenhw/data_science_exam_24.git
cd data_science_exam_24
```
> **Step 2** Run ```setup.sh```

To run the program, we have included a bash script that automatically

1. Creates a virtual environment for the project
2. Activates the virtual environment
3. Installs the correct versions of the packages required

Run the code below in your bash terminal:

```bash
bash setup.sh
```

> **Step 3** Run the notebook of your choice

Note: *Make sure you are in the virtual environment before doing so (`venv_NLP_exam`). To activate the virtual environment, make sure to have run setup.sh first. After this, you can manually activate the environment by inserting this into the terminal:*

```bash
source venv_data_science_exam/bin/activate
```

As said introductory, the repository contains the notebooks for replicating the results from the five different methods: ARIMA, NeuralProphet, LSTM, LagLlama, and TimeGPT located in respective folders.

**ARIMA**

* To replicate the results for the ARIMA model, run the following in the terminal, step by step.
```bash
cd ARIMA/
python3 arima_tune.py
python3 arima_test.py
```

**NeuralProphet**
* To replicate the results for the NeuralProphet method, run the following in the terminal, line by line.
```bash
cd NeuralProphet/
python3 neuralprophet_tune.py
python3 neuralprophet_test.py
```

**RNN (LSTM)**
* To replicate the results for the LSTM model, run the following in the terminal, line by line.
```bash
cd RNN/
python3 RNN_analysis.py
```
* To generate the error metrics, run the notebook ```error_metrics.ipynb```.

**LagLlama**
* NOTE: LagLlama requires some extra steps to run efficiently. Read and run the ```lagllama_setup.sh``` in your terminal and then the other steps:
```bash
cd LagLlama/
bash lagllama_setup.sh
python3 lag_llama_tune.py
python3 lag_llama_test.py
```

**TimeGPT**

* Unlike the other models, TimeGPT are closed source and can only be used with an API key. This key can be obtained by registering on https://docs.nixtla.io/. Place the API key in a .env file, in which you add the line ```NIXTLATS_API_KEY = "your-API-key-goes-here"```.
* To replicate the results for TimeGPT, run the following in the terminal, line by line.
```bash
cd TimeGPT/
python3 timeGPT_analysis.py
```
* To generate the error metrics, run the content of the notebook ```error_metrics.ipynb```.

## Repository structure
```
.
├── .gitignore
├── ARIMA
│   ├── arima_test.py
│   ├── arima_tune.py
│   ├── outputs
│   └── utils.py
├── LagLlama
│   ├── lag_llama_test.py
│   ├── lag_llama_tune.py
│   ├── lagllama_setup.sh
│   ├── outputs
│   └── utils
│       └── lag_utils.py
├── NeuralProphet
│   ├── neuralprophet_test.py
│   ├── neuralprophet_tune.py
│   ├── outputs
│   ├── readme.md
│   └── utils.py
├── README.md
├── RNN
│   ├── analysis_climate.ipynb
│   ├── analysis_weather.ipynb
│   ├── error_metrics.ipynb
│   └── results
│       ├── all_error_metrics
│       ├── climate_data
│       └── weather_data
├── TimeGPT
│   ├── analysis_climate.ipynb
│   ├── analysis_weather.ipynb
│   ├── error_metrics.ipynb
│   └── results
│       ├── all_error_metrics
│       ├── climate_data
│       └── weather_data
├── data
│   ├── climate
│   │   ├── AMOCdata.csv
│   │   ├── data_partitioning.ipynb
│   │   ├── readme.md
│   │   └── splits
│   ├── data_visualisation.ipynb
│   ├── figures
│   ├── splitting_datasets.ipynb
│   └── weather
│       ├── data_partitioning.ipynb
│       └── splits
├── requirements.txt
├── setup.sh
└── src
    ├── forecast_comparison.png
    ├── result_plots.ipynb
    ├── rolling_origin_func.py
    └── visualizations.ipynb
```

## Results

Below are the results from the analysis (in HTML):

<table>
    <caption>Comparison of models across different datasets and horizons. Error metric: Root Mean Squared Error (RMSE)</caption>
    <tr>
        <th rowspan="3">Data Size</th>
        <th rowspan="3">Model</th>
        <th colspan="4">Datasets</th>
    </tr>
    <tr>
        <th colspan="2"><i>Horizon (10)</i></th>
        <th colspan="2"><i>Horizon (50)</i></th>
    </tr>
    <tr>
        <th><i>Weather</i></th>
        <th><i>Sea Temperature</i></th>
        <th><i>Weather</i></th>
        <th><i>Sea Temperature</i></th>
    </tr>
    <tr>
        <td rowspan="5">100 rows</td>
        <td>ARIMA</td>
        <td>1.271 ± 0.776</td>
        <td>0.145 ± 0.057</td>
        <td>2.589 ± 1.378</td>
        <td>0.187 ± 0.038</td>
    </tr>
    <tr>
        <td>RNN (LSTM)</td>
        <td>1.659 ± 0.981</td>
        <td>0.132 ± 0.029</td>
        <td>3.344 ± 0.986</td>
        <td><b>0.178 ± 0.032</b></td>
    </tr>
    <tr>
        <td>NeuralProphet</td>
        <td>1.339 ± 0.434</td>
        <td>0.194 ± 0.050</td>
        <td>7.285 ± 5.134</td>
        <td>0.300 ± 0.071</td>
    </tr>
    <tr>
        <td>LagLlama</td>
        <td>1.351 ± 0.721</td>
        <td><b>0.131 ± 0.035</b></td>
        <td>2.712 ± 1.379</td>
        <td>0.188 ± 0.035</td>
    </tr>
    <tr>
        <td>TimeGPT</td>
        <td><b>0.927 ± 0.529</b></td>
        <td>0.179 ± 0.059</td>
        <td><b>2.247 ± 1.356</b></td>
        <td>0.275 ± 0.123</td>
    </tr>
    <tr>
        <td rowspan="5">1000 rows</td>
        <td>ARIMA</td>
        <td>2.069 ± 1.260</td>
        <td><b>0.153 ± 0.038</b></td>
        <td><b>2.415 ± 0.497</b></td>
        <td>0.196 ± 0.060</td>
    </tr>
    <tr>
        <td>RNN (LSTM)</td>
        <td>1.254 ± 0.792</td>
        <td>0.168 ± 0.059</td>
        <td>4.611 ± 1.52</td>
        <td><b>0.189 ± 0.055</b></td>
    </tr>
    <tr>
        <td>NeuralProphet</td>
        <td>1.302 ± 0.607</td>
        <td>0.191 ± 0.058</td>
        <td>4.273 ± 1.661</td>
        <td>0.351 ± 0.062</td>
    </tr>
    <tr>
        <td>LagLlama</td>
        <td>2.212 ± 1.051</td>
        <td>0.172 ± 0.056</td>
        <td>2.782 ± 1.027</td>
        <td>0.194 ± 0.057</td>
    </tr>
    <tr>
        <td>TimeGPT</td>
        <td><b>1.164 ± 0.497</b></td>
        <td>0.181 ± 0.065</td>
        <td>2.705 ± 0.991</td>
        <td>0.211 ± 0.074</td>
    </tr>
</table>

The table presents the mean RMSE and standard deviation for each model across the different datasets, forecast horizons and training data sizes. In general, all models perform comparably to each other. Overall TimeGPT has the best performance in 4 of the possible 8, RNN has the best performance in 2, while Lag-Llama and Arima both have one. But for most of these only small differences set them apart. NeuralProphet seems to be the worst-performing model for most datasets and in most scenarios.