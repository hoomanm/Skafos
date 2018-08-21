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
	n_vars = len(data.columns)
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
	result = pd.concat(cols, axis=1)
	result.columns = names

	# drop rows with NaN values
	if dropnan:
		result.dropna(inplace=True)
	return result



NUM_TIME_STEPS = 5

df = pd.read_csv("coffee_2007-2018.csv")

# Drop the date column
df = df.drop('date', axis=1)

NUM_FEATURES = len(df.columns)

# scale features
scaler = MinMaxScaler(feature_range=(-1, 1))
df[df.columns] = scaler.fit_transform(df[df.columns])

# frame as supervised learning
supervised_data = series_to_supervised(df, NUM_TIME_STEPS)

# split into train and test sets
num_training_examples = int(supervised_data.shape[0] * 0.7)
supervised_data_values = supervised_data.values
train_data = supervised_data_values[:num_training_examples, :]
test_data = supervised_data_values[num_training_examples:, :]

# split into input and outputs
train_X = train_data[:, :NUM_TIME_STEPS * NUM_FEATURES]
train_y = train_data[:, -1]

test_X = test_data[:, :NUM_TIME_STEPS * NUM_FEATURES]
test_y = test_data[:, -1]

# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], NUM_TIME_STEPS, NUM_FEATURES))
test_X = test_X.reshape((test_X.shape[0], NUM_TIME_STEPS, NUM_FEATURES))

# build LSTM network
lstm_model = Sequential()
lstm_model.add(LSTM(50, batch_input_shape=(1, train_X.shape[1], train_X.shape[2]), stateful=True))
lstm_model.add(Dense(1))
lstm_model.compile(loss='mae', optimizer='adam')

# fit network
for i in range(200):
	lstm_model.fit(train_X, train_y, epochs=1, batch_size=1, verbose=2, shuffle=False)
	lstm_model.reset_states()


# walk-forward validation on the test data
predictions = list()
true_values = list()
for i in range(len(test_X)):

	# make one-step forecast
	test_X_i = test_X[i,:].reshape(1, NUM_TIME_STEPS, NUM_FEATURES)
	predicted_y = lstm_model.predict(test_X_i, batch_size=1)
	
	# invert scaling
	inv_predicted_y = np.concatenate((test_data[i, -NUM_FEATURES:-1].reshape(1, NUM_FEATURES-1), predicted_y), axis=1)
	inv_predicted_y = scaler.inverse_transform(inv_predicted_y.reshape(1,-1))
	inv_predicted_y = inv_predicted_y[:,-1]

	test_y = test_y.reshape((len(test_y), 1))
	test_row = np.concatenate((test_data[i, -NUM_FEATURES:-1], test_y[i]))
	inv_test = scaler.inverse_transform(test_row.reshape(1,-1))
	inv_y = inv_test[:,-1]
	
	# store forecast and true values
	predictions.append(inv_predicted_y)
	true_values.append(inv_y)
	#print('Day=%d, Predicted=%f, Expected=%f' % (i+1, inv_predicted_y, inv_y))


# evaluate model
rmse = sqrt(mean_squared_error(true_values, predictions))
mae = mean_absolute_error(true_values, predictions)
print('Test RMSE: %.3f' % rmse)
print('Test MAE: %.3f' % mae)




