#!/bin/bash

# problems running on python 3.12
# recommend running on python 3.10.0 with pyenv
# initialise pyenv before running this script

cd LagLlama

git clone https://github.com/time-series-foundation-models/lag-llama.git

cp lag_llama_tune.py lag-llama/
cp lag_llama_test.py lag-llama/
cp -r outputs lag-llama/
cp utils/lag_utils.py lag-llama/utils/

cd lag-llama

pip install -r requirements.txt

huggingface-cli download time-series-foundation-models/Lag-Llama lag-llama.ckpt --local-dir lag-llama