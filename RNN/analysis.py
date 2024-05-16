# data wrangling, plotting
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# RNN libraries
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import SimpleRNN
from keras.layers import Dropout
from keras.layers import GRU, Bidirectional
from keras.optimizers import SGD
from sklearn import metrics
from sklearn.metrics import mean_squared_error

# test/training small
df_train_small = pd.read_csv("../data/weather/AarhusSydObservations/splits/train/train_small.csv")
df_test_small = pd.read_csv("../data/weather/AarhusSydObservations/splits/test/test_small.csv")
# test/training large
df_train_large = pd.read_csv("../data/weather/AarhusSydObservations/splits/train/train_large.csv")
df_test_large = pd.read_csv("../data/weather/AarhusSydObservations/splits/test/test_large.csv")

def preprocess_data(train, test, column='Middeltemperatur'):
    # Drop unnecessary columns
    columns_to_drop = ['Luftfugtighed', 'Nedbør', 'Nedbørsminutter','Maksimumtemperatur', 'Minimumtemperatur', 'Skyhøjde', 'Skydække', 'Middelvindhastighed', 'Højeste vindstød']
    train = train.drop(columns=columns_to_drop)
    test = test.drop(columns=columns_to_drop)

    # Convert 'DateTime' to datetime and set as index
    train['DateTime'] = pd.to_datetime(train['DateTime'])
    train = train.set_index('DateTime')

    test['DateTime'] = pd.to_datetime(test['DateTime'])
    test = test.set_index('DateTime')

    dataset_train = train[column].values 
    dataset_train = np.reshape(dataset_train, (-1,1))

    # Selecting column values
    dataset_test = test[column].values 
    # Reshaping 1D to 2D array
    dataset_test = np.reshape(dataset_test, (-1,1))
    

    return dataset_train, dataset_test

train_small, test_small = preprocess_data(df_train_small, df_test_small)
train_large, test_large = preprocess_data(df_train_large, df_test_large)

####### MIN MAX SCALING DATA #########

def scale_data(train, test):
    # Create scaler object
    scaler = MinMaxScaler(feature_range=(0,1))
    # scaling dataset
    scaled_train = scaler.fit_transform(train)
    scaled_test = scaler.fit_transform(test) 
    return scaled_train, scaled_test, scaler

scaled_train_small, scaled_test_small, scaler = scale_data(train_small, test_small)
scaled_train_large, scaled_test_large, scaler = scale_data(train_large, test_large)


def create_sequences(scaled_data, look_back=24, predict_forward=12):
    X, y = [], []
    for i in range(look_back, len(scaled_data) - predict_forward):
        X.append(scaled_data[i-look_back:i, 0])
        y.append(scaled_data[i + predict_forward, 0])  # Predict the point 12 steps ahead

    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    y = y.reshape((y.shape[0], 1))
    
    return X, y

X_train_small, y_train_small = create_sequences(scaled_train_small)
X_train_large, y_train_large = create_sequences(scaled_train_large)

X_test_small, y_test_small = create_sequences(scaled_test_small)
X_test_large, y_test_large = create_sequences(scaled_test_large)

############################
####### RNN MODEL ##########
############################
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

regressor = build_rnn((X_train_large.shape[1], 1))

fitted_model = fit_regressor(regressor, X_train_large, y_train_large, epochs=1, batch_size=1)

############################
#### ROLLING ORIGIN PREDICTIONS #####
############################

# Generating rolling origin predictions
window_size = 24
predict_ahead = 12
rolling_predictions = []

############################
#### ROLLING ORIGIN PREDICTIONS #####
############################

def rolling_origin_predictions(model, scaled_test, scaler, look_back=24, predict_forward=12):
    predictions = []
    for start in range(len(scaled_test) - look_back - predict_forward):
        end = start + look_back
        X_test = scaled_test[start:end]

        # reshape input to be [samples, time steps, features]
        X_test = np.reshape(X_test, (1, X_test.shape[0], 1))

        # Make prediction for the next 'predict_forward' steps
        preds = model.predict(X_test) # Get the prediction for the next step
        preds = scaler.inverse_transform(preds) # Inverse transform to original scale
        predictions.append(preds[0][0])
    
    return predictions

# Generate rolling origin predictions
rolling_predictions_small = rolling_origin_predictions(fitted_model, scaled_test_small, scaler)
rolling_predictions_large = rolling_origin_predictions(fitted_model, scaled_test_large, scaler)

# Plotting the results
def plot_results(rolling_predictions, test_data, train_data, title):
    fig, axs = plt.subplots(1, figsize=(18, 6))
    fig.suptitle(title)

    # Plot train and test data
    axs.plot(train_data.index, train_data['Middeltemperatur'], label="train_data", color="blue")
    axs.plot(test_data.index, test_data['Middeltemperatur'], label="test_data", color="green")

    # Calculate indices for rolling predictions
    predict_start_idx = train_data.shape[0] + look_back
    predict_indices = test_data.index[look_back:-predict_forward]

    axs.plot(predict_indices, rolling_predictions, label="rolling_predictions", color="brown")
    axs.legend()
    axs.set_title(title)

    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.show()

# Plot for small dataset
plot_results(rolling_predictions_small, df_test_small, df_train_small, "RNN with Rolling Predictions (Small Dataset)")

# Plot for large dataset
plot_results(rolling_predictions_large, df_test_large, df_train_large, "RNN with Rolling Predictions (Large Dataset)")