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
        y.append(scaled_data[i + predict_forward, 0])  # Predict the point 12 steps ahead

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
    axs[0].plot(test.index[30:-12], y_pred, label="y_RNN", color="brown")  # Subtract 12 from the end
    axs[0].legend()
    axs[0].title.set_text("Basic RNN")

    # Saving plot
    plt.savefig("RNN_predictions.png")

import pandas as pd

def rolling_origin_predict(regressor, scaled_train, scaled_test, scaler, look_back=30, predict_forward=12):
    predictions = []

    new_test_set = np.concatenate((scaled_train[-look_back:], scaled_test), axis=0)

    for i in range(look_back, len(new_test_set) - predict_forward):
        X = new_test_set[i-look_back:i, 0]
        X = X.reshape((1, X.shape[0], 1))
        y = regressor.predict(X)
        predictions.append(y[0][:predict_forward])  # Predict 12 points forward

    # Convert predictions to a 2D array
    predictions = np.array(predictions)

    # Inverse transform each column
    for i in range(predictions.shape[1]):
        predictions[:, i] = scaler.inverse_transform(predictions[:, i].reshape(-1, 1)).flatten()

    # Convert to DataFrame
    df_predictions = pd.DataFrame(predictions, columns=[f'roll_{i+1}' for i in range(predict_forward)])

    # Save to csv
    df_predictions.to_csv("RNN_predictions.csv", index=False)

    return df_predictions