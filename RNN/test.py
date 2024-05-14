import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

# reading train set (120 len test, 750 len train)
train = pd.read_csv("../data/weather/AarhusSydObservations/data_partitions/Partition_test_120_train_750_train.csv")
test = pd.read_csv("../data/weather/AarhusSydObservations/data_partitions/Partition_test_120_train_750_test.csv")

test = test.drop(columns=['Luftfugtighed', 'Nedbør', 'Nedbørsminutter','Maksimumtemperatur', 'Minimumtemperatur', 'Skyhøjde', 'Skydække', 'Middelvindhastighed', 'Højeste vindstød'])
train = train.drop(columns=['Luftfugtighed', 'Nedbør', 'Nedbørsminutter','Maksimumtemperatur', 'Minimumtemperatur', 'Skyhøjde', 'Skydække', 'Middelvindhastighed', 'Højeste vindstød'])

data = pd.concat([train, test])

############################
####### PREPROCESSING ######
############################

# make DateTime the index
data['DateTime'] = pd.to_datetime(data['DateTime'])
data = data.set_index('DateTime')

# Selecting middeltemperatur values
dataset_train = data.Middeltemperatur.values 
# Reshaping 1D to 2D array
dataset_train = np.reshape(dataset_train, (-1,1)) 

####### MIN MAX SCALING DATA #########

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))
scaled_train = scaler.fit_transform(dataset_train) # Fit to training data


# Create sequences and labels for training (24 hours)
X = []
y = []
for i in range(24, len(scaled_train)):
	X.append(scaled_train[i-24:i, 0])
	y.append(scaled_train[i, 0])

# The TRAIN data is converted to Numpy array
X, y = np.array(X), np.array(y)

# splitting the data into train and test
train_size = int(0.8 * len(X)) # 80% train, 20% test
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1],1))
y_train = np.reshape(y_train, (y_train.shape[0],1))
print("X_train :", X_train.shape,"y_train :", y_train.shape)

# The TEST data is converted to numpy array
X_test, y_test = np.array(X_test), np.array(y_test)

#Reshaping
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],1))
y_test = np.reshape(y_test, (y_test.shape[0],1))
print("X_test :",X_test.shape,"y_test :",y_test.shape)

############################
####### RNN MODEL ##########
############################

# importing libraries
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import SimpleRNN
from keras.layers import Dropout
from keras.layers import GRU, Bidirectional
from keras.optimizers import SGD
from sklearn import metrics
from sklearn.metrics import mean_squared_error

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
regressor.fit(X_train, y_train, epochs = 20, batch_size = 2)
regressor.summary()

# predictions with X_test data
y_RNN = regressor.predict(X_test)

# scaling back from 0-1 to original
y_RNN_O = scaler.inverse_transform(y_RNN) 

# Scaling back y_test to original range
y_test_O = scaler.inverse_transform(y_test)

# Plotting the results
plt.figure(figsize=(14,5))
plt.plot(y_test_O, color = 'red', label = 'Real temperature')
plt.plot(y_RNN_O, color = 'blue', label = 'Predicted temperature')
plt.title('Temperature Prediction')
plt.xlabel('Time')
plt.ylabel('Temperature')
plt.legend()
plt.show()

# save the plot
plt.savefig("RNN_plot.png")