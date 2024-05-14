import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

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


# reading train set (120 len test, 750 len train)
train = pd.read_csv("../data/weather/AarhusSydObservations/data_partitions/Partition_test_120_train_750_train.csv")
test = pd.read_csv("../data/weather/AarhusSydObservations/data_partitions/Partition_test_120_train_750_test.csv")

test = test.drop(columns=['Luftfugtighed', 'Nedbør', 'Nedbørsminutter','Maksimumtemperatur', 'Minimumtemperatur', 'Skyhøjde', 'Skydække', 'Middelvindhastighed', 'Højeste vindstød'])
train = train.drop(columns=['Luftfugtighed', 'Nedbør', 'Nedbørsminutter','Maksimumtemperatur', 'Minimumtemperatur', 'Skyhøjde', 'Skydække', 'Middelvindhastighed', 'Højeste vindstød'])


############################
####### PREPROCESSING ######
############################

# make DateTime the index
train['DateTime'] = pd.to_datetime(train['DateTime'])
train = train.set_index('DateTime')
# make DateTime the index
test['DateTime'] = pd.to_datetime(test['DateTime'])
test = test.set_index('DateTime')

# Selecting middeltemperatur values
dataset_train = train.Middeltemperatur.values 
# Reshaping 1D to 2D array
dataset_train = np.reshape(dataset_train, (-1,1)) 

# Selecting Open Price values
dataset_test = test.Middeltemperatur.values 
# Reshaping 1D to 2D array
dataset_test = np.reshape(dataset_test, (-1,1)) 

####### MIN MAX SCALING DATA #########

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))
scaled_train = scaler.fit_transform(dataset_train) # Fit to training data
scaled_test = scaler.transform(dataset_test)       # Transform test data with same scaler


# Creating training sequences
X_train = []
y_train = []
for i in range(24, len(scaled_train)-12):  # Subtract 12 to avoid going out of bounds
    X_train.append(scaled_train[i-24:i, 0])
    y_train.append(scaled_train[i+12, 0])  # Predict the point 12 steps ahead

# Creating testing sequences
X_test = []
y_test = []
for i in range(24, len(scaled_test)-12):  # Subtract 12 to avoid going out of bounds
    X_test.append(scaled_test[i-24:i, 0])
    y_test.append(scaled_test[i+12, 0])  # Predict the point 12 steps ahead

# Convert to Numpy arrays and reshape
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
y_train = np.reshape(y_train, (y_train.shape[0], 1))

X_test, y_test = np.array(X_test), np.array(y_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
y_test = np.reshape(y_test, (y_test.shape[0], 1))

print("X_train :", X_train.shape,"y_train :", y_train.shape)
print("X_test :",X_test.shape,"y_test :",y_test.shape)

############################
####### RNN MODEL ##########
############################

# initializing the RNN
regressor = Sequential()

# adding RNN layers and dropout regularization
regressor.add(SimpleRNN(units = 50,
						activation = "tanh",
						return_sequences = True,
						input_shape = (X_train.shape[1],1)))

#regressor.add(Dropout(0.2))

regressor.add(SimpleRNN(units = 50, 
						activation = "tanh",
						return_sequences = True))

regressor.add(SimpleRNN(units = 50,
						activation = "tanh",
						return_sequences = True))

regressor.add( SimpleRNN(units = 50))

# adding the output layer
regressor.add(Dense(units = 1,activation='sigmoid'))

# compiling RNN
regressor.compile(optimizer = SGD(learning_rate=0.01, # learning rate
								decay=1e-6, # 
								momentum=0.9, 
								nesterov=True), 
				loss = "mean_squared_error")

# fitting the model
regressor.fit(X_train, y_train, epochs = 10, batch_size = 1)
regressor.summary()

# predictions with X_test data
y_RNN = regressor.predict(X_test)

# scaling back from 0-1 to original
y_RNN_O = scaler.inverse_transform(y_RNN) 

fig, axs = plt.subplots(3,figsize =(18,12),sharex=True, sharey=True)
fig.suptitle('Model Predictions')

#Plot for RNN predictions
axs[0].plot(train.index[150:], train.Middeltemperatur[150:], label = "train_data", color = "b")
axs[0].plot(test.index, test.Middeltemperatur, label = "test_data", color = "g")
axs[0].plot(test.index[24:-12], y_RNN_O, label = "y_RNN", color = "brown")  # Subtract 12 from the end
axs[0].legend()
axs[0].title.set_text("Basic RNN")

# saving plot
plt.savefig("RNN_predictions.png")
