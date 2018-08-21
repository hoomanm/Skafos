import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg


NUM_TIME_STEPS = 30

df = pd.read_csv("coffee_2007-2018.csv")

# Drop the date column
df = df.drop('date', axis=1)

NUM_FEATURES = len(df.columns)
print(df.describe())
# print(NUM_FEATURES)

original_data = df.values
# print(data)
# print(data.shape)

# normalize features
scaler = MinMaxScaler(feature_range=(-1, 1))
scaled_data = scaler.fit_transform(original_data)
# print(scaled_data)

# frame as supervised learning
supervised_data = series_to_supervised(scaled_data, NUM_TIME_STEPS)

# print(df.head())
# print(df.describe())
# print(supervised_data.head())
# print(supervised_data.describe())

# drop columns we don't want to predict
# supervised_data.drop(supervised_data.columns[[12,13,14]], axis=1, inplace=True)
print(supervised_data.head())
print(supervised_data.describe())

# split into train and test sets
supervised_data_values = supervised_data.values
num_training_examples = int(supervised_data_values.shape[0] * 0.7)
#print("Training Examplessssss:", num_training_examples)
train_data = supervised_data_values[:num_training_examples, :]
test_data = supervised_data_values[num_training_examples:, :]

# split into input and outputs
train_X = train_data[:, :NUM_TIME_STEPS * NUM_FEATURES]
train_y = train_data[:, -1]

test_X = test_data[:, :NUM_TIME_STEPS * NUM_FEATURES]
test_y = test_data[:, -1]

print(test_X.shape)
print(test_y.shape)

# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], NUM_TIME_STEPS, NUM_FEATURES))
test_X = test_X.reshape((test_X.shape[0], NUM_TIME_STEPS, NUM_FEATURES))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# design network
lstm_model = Sequential()
lstm_model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
lstm_model.add(Dense(1))
lstm_model.compile(loss='mae', optimizer='adam')
# fit network
history = lstm_model.fit(train_X, train_y, epochs=200, batch_size=100, validation_data=(test_X, test_y), verbose=2, shuffle=False)
# plot history
#plt.plot(history.history['loss'], label='train')
#plt.plot(history.history['val_loss'], label='test')
#plt.legend()
#plt.show()

# make a prediction
predicted_y = lstm_model.predict(test_X, batch_size=test_X.shape[0])
# print(predicted_y.shape)
# test_X = test_X.reshape((test_X.shape[0], NUM_TIME_STEPS * NUM_FEATURES))
# invert scaling for forecast
inv_predicted_y = np.concatenate((test_data[:, -NUM_FEATURES:-1], predicted_y), axis=1)
inv_predicted_y = scaler.inverse_transform(inv_predicted_y)
inv_predicted_y = inv_predicted_y[:,-1]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = np.concatenate((test_data[:, -NUM_FEATURES:-1], test_y), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,-1]
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_predicted_y))
mae = mean_absolute_error(inv_y, inv_predicted_y)
print('Test RMSE: %.3f' % rmse)
print('Test MAE: %.3f' % mae)


#df.close.plot()
#plt.show()











