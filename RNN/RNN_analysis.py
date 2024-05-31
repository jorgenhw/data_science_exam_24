import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import kerastuner as kt

# Load and preprocess CLIMATE data
df_all_data = pd.read_csv("../data/climate/AMOCdata.csv")

def preprocess_data(df):
    columns_to_drop = ['AMOC1', 'AMOC2', 'GM', 'time']
    df = df.drop(columns=columns_to_drop)
    return df

all_data = preprocess_data(df_all_data)
all_data['DateTime'] = pd.to_datetime(all_data['date'])
all_data = all_data.drop(columns=['date'])


# load and preprocess WEATHER data (uncomment below line to use the weather data instead)
#df_all_data = pd.read_csv("../data/weather/AarhusSydObservations/weatherAarhusSyd_Marts_april.csv")

#def preprocess_data(df, column='Middeltemperatur'):
#    # Drop unnecessary columns
#    columns_to_drop = ['Luftfugtighed', 'Nedbør', 'Nedbørsminutter','Maksimumtemperatur', 'Minimumtemperatur', 'Skyhøjde', 'Skydække', 'Middelvindhastighed', 'Højeste vindstød']
#    df = df.drop(columns=columns_to_drop)
    
#    return df

#all_data = preprocess_data(df_all_data)


# Set index
df = all_data.set_index('DateTime')

# Normalize the dataset
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

# Split into training and test sets
train_size = 100
test_size = 250
predictions_ahead = 50

train = scaled_data[:train_size]
test = scaled_data[train_size:train_size + test_size]

# Extract datetime for the test set
test_dates = df.index[train_size:train_size + test_size]

# Function to create dataset
def create_dataset(dataset, look_back=50):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

look_back = 50
trainX, trainY = create_dataset(train, look_back)

# Reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))

# Split trainX and trainY for training and validation
val_size = int(trainX.shape[0] * 0.2)
valX, valY = trainX[:val_size], trainY[:val_size]
trainX, trainY = trainX[val_size:], trainY[val_size:]

# Build the model - Hyperparameter tuning
def build_model(hp):
    model = Sequential()
    model.add(LSTM(hp.Int('units_1', min_value=32, max_value=128, step=32), return_sequences=True, input_shape=(look_back, 1)))
    model.add(Dropout(hp.Float('dropout_1', min_value=0.0, max_value=0.5, step=0.1)))
    model.add(LSTM(hp.Int('units_2', min_value=32, max_value=128, step=32)))
    model.add(Dropout(hp.Float('dropout_2', min_value=0.0, max_value=0.5, step=0.1)))
    model.add(Dense(1))
    model.compile(optimizer=hp.Choice('optimizer', values=['adam', 'rmsprop']),
                  loss='mean_squared_error')
    return model

# Instantiate the tuner
tuner = kt.Hyperband(build_model,
                     objective='val_loss',
                     max_epochs=50,
                     hyperband_iterations=2)

# Normalize the dataset
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))

# Search for the best hyperparameters
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
tuner.search(trainX, trainY, epochs=50, validation_data=(valX, valY), callbacks=[early_stop])

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters()[0]

# Build the model with the optimal hyperparameters
model = tuner.hypermodel.build(best_hps)

# Train the model
history = model.fit(trainX, trainY, epochs=50, validation_data=(valX, valY), callbacks=[early_stop], verbose=2)

# Function to predict and move with a rolling window (with one timestep increment)
def rolling_window_prediction(model, train_data, test_data, test_dates, look_back, num_predictions):
    predictions = []
    actuals = []
    dates = []
    roll_numbers = []
    roll_number = 0
    full_data = np.concatenate((train_data, test_data))
    full_dates = np.concatenate((df.index[:train_size], test_dates))
    start_index = len(train_data) - look_back

    for i in range(start_index, len(full_data) - look_back - num_predictions + 1):
        roll_number += 1
        input_data = full_data[i:i + look_back].reshape((1, look_back, 1))

        for step in range(num_predictions):
            if i + look_back + step >= len(full_data):
                break

            predicted = model.predict(input_data)
            predictions.append(predicted.flatten()[0])
            actuals.append(full_data[i + look_back + step, 0])
            dates.append(full_dates[i + look_back + step])
            roll_numbers.append(roll_number)
            input_data = np.append(input_data[:, 1:], predicted.reshape(1, 1, 1), axis=1)

    return np.array(predictions), np.array(actuals), dates, roll_numbers

# Execute rolling prediction function
preds, trues, preds_dates, roll_numbers = rolling_window_prediction(model, train, test, test_dates, look_back, num_predictions=predictions_ahead)

# Convert lists to pandas DataFrame
results_df = pd.DataFrame({
    'DateTime': preds_dates,
    'Predicted': scaler.inverse_transform(preds.reshape(-1, 1)).flatten(),
    'Actual': scaler.inverse_transform(trues.reshape(-1, 1)).flatten(),
    'RollNumber': roll_numbers
})
results_df.set_index('DateTime', inplace=True)

print(results_df.head())
print('Number of predictions:', len(results_df))

# Save results to CSV
results_df.to_csv(f'outputs/climate_data/predictions/train_{train_size}_test_{test_size}_horizon_{predictions_ahead}_results.csv')

# save weather results to csv
#results_df.to_csv(f'outputs/weather_data/predictions/train_{train_size}_test_{test_size}_horizon_{predictions_ahead}_results.csv')