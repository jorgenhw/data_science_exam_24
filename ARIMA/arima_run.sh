#!/bin/bash

echo "ARIMA running"

# Run ARIMA for all combinations of train and test size
python ARIMA/arima_script.py 0 0
# python ARIMA/arima_script.py 0 1
# python ARIMA/arima_script.py 0 2
# python ARIMA/arima_script.py 1 0
# python ARIMA/arima_script.py 1 1
# python ARIMA/arima_script.py 1 2
# python ARIMA/arima_script.py 2 0
# python ARIMA/arima_script.py 2 1
# python ARIMA/arima_script.py 2 2

echo "ARIMA done"