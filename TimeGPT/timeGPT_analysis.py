import os
import pandas as pd
from nixtlats import NixtlaClient
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize Nixtla client with API key from environment variables
nixtla_client = NixtlaClient(
    api_key=os.getenv("NIXTLATS_API_KEY")
)

# Check if the API key is valid
nixtla_client.validate_api_key()

# Load the climate dataset 
data_path = os.path.join('..', 'data', 'climate', 'AMOCdata.csv')
all_data = pd.read_csv(data_path)

# load the weather data (uncomment below line to use the weather data instead)
#all_data = pd.read_csv(os.path.join('..','data','weather', 'AarhusSydObservations', 'weatherAarhusSyd_Marts_april.csv'))

# Define parameters
train_size = 1000
test_size = 250
horizon = 50
model_type = 'timegpt-1-long-horizon'  # 'timegpt-1' or 'timegpt-1-long-horizon'
target_column = 'AMOC0' # or Middeltemperatur (for weather)
time_column = 'date' # or DateTime for the weather dataset
frequency = 'MS'  # month start frequency (H for hourly, which should be used for the weather dataset)

def rolling_origin_forecast(train_size, test_size, horizon, dataset, time_column, target_column, frequency, model_type):
    all_predictions = []
    for start in range(train_size, (train_size + test_size) - horizon + 1):
        train_df = dataset.iloc[:start]

        # Predict the next horizon rows
        forecast_df = nixtla_client.forecast(
            df=train_df,
            h=horizon,
            time_col=time_column,
            target_col=target_column,
            freq=frequency,
            model=model_type,
        )

        # Get the actual values for the prediction window
        actual_df = dataset.iloc[start:start + horizon][[time_column, target_column]]

        # Concatenate the actual and prediction DataFrames
        results = forecast_df.copy()
        results['Actual'] = actual_df[target_column].values

        # Save the results in the list
        all_predictions.append(results)
    
    return all_predictions

# Perform rolling origin forecast
results = rolling_origin_forecast(train_size, test_size, horizon, all_data, time_column, target_column, frequency, model_type)

# Combine all predictions into a single DataFrame
results_df = pd.concat(results)
results_df['RollingOrigin'] = [i for i in range(len(results)) for _ in range(horizon)]

# Save the results to a CSV file
output_path = os.path.join('output', 'climate_data', 'predictions', f'train_{train_size}_test_{test_size}_horizon_{horizon}_modeltype_{model_type}_climate_results.csv')
results_df.to_csv(output_path, index=False)

print(f"Results saved to {output_path}")