# Imports
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense
from keras.optimizers import SGD
from sklearn.preprocessing import MinMaxScaler
from typing import List, Tuple

def load_data(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    columns_to_drop = ['Luftfugtighed', 'Nedbør', 'Nedbørsminutter',
                       'Maksimumtemperatur', 'Minimumtemperatur', 'Skyhøjde', 'Skydække', 
                       'Middelvindhastighed', 'Højeste vindstød']
    train.drop(columns=columns_to_drop, inplace=True)
    test.drop(columns=columns_to_drop, inplace=True)

    return train, test

def preprocess_data(train, test):
    # Make DateTime the index
    train['DateTime'] = pd.to_datetime(train['DateTime'])
    train.set_index('DateTime', inplace=True)

    test['DateTime'] = pd.to_datetime(test['DateTime'])
    test.set_index('DateTime', inplace=True)

    # Selecting temperature values and reshaping to 2D
    dataset_train = train['Middeltemperatur'].values.reshape(-1, 1)
    dataset_test = test['Middeltemperatur'].values.reshape(-1, 1)
    
    return dataset_train, dataset_test

def scale_data(dataset_train, dataset_test):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train = scaler.fit_transform(dataset_train)
    scaled_test = scaler.transform(dataset_test)

    return scaled_train, scaled_test, scaler

def create_sequences(scaled_data, look_back=30, predict_forward=12):
    X, y = [], []
    for i in range(look_back, len(scaled_data) - predict_forward):
        X.append(scaled_data[i-look_back:i, 0])
        y.append(scaled_data[i + predict_forward, 0])

    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    y = y.reshape((y.shape[0], 1))
    
    return X, y

def build_rnn(input_shape):
    regressor = Sequential()

    regressor.add(SimpleRNN(units=50, activation="tanh", return_sequences=True, input_shape=input_shape))
    regressor.add(SimpleRNN(units=50, activation="tanh", return_sequences=True))
    regressor.add(SimpleRNN(units=50, activation="tanh", return_sequences=True))
    regressor.add(SimpleRNN(units=50))
    regressor.add(Dense(units=1, activation='sigmoid'))

    optimizer = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    regressor.compile(optimizer=optimizer, loss="mean_squared_error")

    return regressor

def fit_regressor(regressor, X_train, y_train, epochs=20, batch_size=1):
    regressor.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
    return regressor

def plot_predictions(train, test, y_pred):
    fig, axs = plt.subplots(3, figsize=(18, 12), sharex=True, sharey=True)
    fig.suptitle('Model Predictions')

    # Plot for RNN predictions
    axs[0].plot(train.index[150:], train['Middeltemperatur'][150:], label="train_data", color="b")
    axs[0].plot(test.index, test['Middeltemperatur'], label="test_data", color="g")
    axs[0].plot(test.index[0:len(y_pred)], y_pred, label="y_RNN", color="brown")  # Update to match predictions
    axs[0].legend()
    axs[0].title.set_text("Basic RNN")

    # Saving plot
    plt.savefig("RNN_predictions.png")

def rolling_origin_predict(regressor, train, test, scaled_train, scaled_test, look_back=30, predict_forward=12):
    predictions = []
    timestamps = []

    # Initial input_data combining the last `look_back` points from train
    # and extending progressively from test
    input_data = np.concatenate((scaled_train[-look_back:], scaled_test[:0]), axis=0).reshape((1, look_back, 1))

    # Loop over the test set in steps of `predict_forward`
    for start_idx in range(0, len(scaled_test) - predict_forward + 1, predict_forward):
        
        # Predict the next `predict_forward` values
        pred = regressor.predict(input_data)
        predictions.extend(pred.flatten())
        
        # Extract the corresponding timestamps
        timestamps.extend(test.index[start_idx:start_idx + predict_forward].tolist())

        # Update input_data with new points from test for the next rolling prediction
        if start_idx + predict_forward < len(scaled_test):
            next_input = scaled_test[start_idx:start_idx + predict_forward].reshape(-1, 1)
            input_data = np.concatenate((input_data[0, predict_forward:], next_input), axis=0).reshape((1, look_back, 1))
    
    # Ensure predictions match the length of test set
    return np.array(predictions[:len(test)]), timestamps[:len(test)]

def main():
    train_path = "../data/weather/AarhusSydObservations/data_partitions/Partition_test_120_train_750_train.csv"
    test_path = "../data/weather/AarhusSydObservations/data_partitions/Partition_test_120_train_750_test.csv"

    # Load data
    train, test = load_data(train_path, test_path)
    
    # Preprocess data
    dataset_train, dataset_test = preprocess_data(train, test)
    
    # Scale data
    scaled_train, scaled_test, scaler = scale_data(dataset_train, dataset_test)

    # Create sequences (for initial training)
    X_train, y_train = create_sequences(scaled_train, look_back=30, predict_forward=12)
    
    # Build and fit RNN model
    regressor = build_rnn((X_train.shape[1], 1))
    regressor = fit_regressor(regressor, X_train, y_train, epochs=2, batch_size=1)
    regressor.summary()

    # Perform rolling origin predictions
    y_rolling_origin_scaled, timestamps = rolling_origin_predict(regressor, train, test, scaled_train, scaled_test, look_back=30, predict_forward=12)
    y_rolling_origin = scaler.inverse_transform(y_rolling_origin_scaled.reshape(-1, 1))

    # Save predictions to a dataframe
    predictions_df = pd.DataFrame({
        'DateTime': timestamps,
        'Predicted_Temperature': y_rolling_origin.flatten()
    })
    predictions_df.set_index('DateTime', inplace=True)
    predictions_df.to_csv('rolling_predictions.csv')

    # Plot and save plot
    plot_predictions(train, test, y_rolling_origin)

if __name__ == "__main__":
    main()