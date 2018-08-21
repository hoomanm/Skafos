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


# Start

NUM_TIME_STEPS = 1

df = pd.read_csv("coffee_2007-2018.csv")

# Drop the date column
df = df.drop('date', axis=1)

NUM_FEATURES = len(df.columns)
print(df.describe())
# print(NUM_FEATURES)

data = df.values
# print(data)
# print(data.shape)

# normalize features
scaler = MinMaxScaler(feature_range=(-1, 1))
scaled_data = scaler.fit_transform(data)
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

# split into train and test sets
data = supervised_data.values
num_training_examples = int(data.shape[0] * 0.7)
#print("Training Examplessssss:", num_training_examples)
train_data = data[:num_training_examples, :]
test_data = data[num_training_examples:, :]

# split into input and outputs
train_X = train_data[:, :NUM_TIME_STEPS * NUM_FEATURES]
train_y = train_data[:, -1]

test_X = test_data[:, :NUM_TIME_STEPS * NUM_FEATURES]
test_y = test_data[:, -1]

# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], NUM_TIME_STEPS, NUM_FEATURES))
test_X = test_X.reshape((test_X.shape[0], NUM_TIME_STEPS, NUM_FEATURES))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# design network
lstm_model = Sequential()
#lstm_model.add(LSTM(75, input_shape=(train_X.shape[1], train_X.shape[2])))
lstm_model.add(LSTM(5, batch_input_shape=(1, train_X.shape[1], train_X.shape[2]), stateful=True))
lstm_model.add(Dense(1))
lstm_model.compile(loss='mae', optimizer='adam')
# fit network
for i in range(500):
	lstm_model.fit(train_X, train_y, epochs=1, batch_size=1, verbose=2, shuffle=False)
	lstm_model.reset_states()
#history = lstm_model.fit(train_X, train_y, epochs=500, batch_size=250, validation_data=(test_X, test_y), verbose=2, shuffle=False)

#plot history
#plt.plot(history.history['loss'], label='train')
#plt.plot(history.history['val_loss'], label='test')
#plt.legend()
#plt.show()

# walk-forward validation on the test data
predictions = list()
true_values = list()
for i in range(len(test_X)):
	# make one-step forecast
	#X, y = test_X[i, 0:-4], test_X[i, -1]
	test_X_i = test_X[i,:].reshape(1, NUM_TIME_STEPS, NUM_FEATURES)
	predicted_y = lstm_model.predict(test_X_i, batch_size=1)
	#predicted_y = predicted_y[:,0]
	#print(predicted_y)
	#print(test_data[i, -NUM_FEATURES:-1])
	#reshaped_test_data = test_data
	
	# invert scaling
	inv_predicted_y = np.concatenate((test_data[i, -NUM_FEATURES:-1].reshape(1, NUM_FEATURES-1), predicted_y), axis=1)
	inv_predicted_y = scaler.inverse_transform(inv_predicted_y.reshape(1,-1))
	inv_predicted_y = inv_predicted_y[:,-1]

	test_y = test_y.reshape((len(test_y), 1))
	#print(test_y[i])
	#test_row = test_data[i, -NUM_FEATURES:-1]
	#test_row.append(test_y[i])
	#print(test_row)
	#test_row = test_row.reshape(1, NUM_FEATURES)
	#print(test_data[i, -NUM_FEATURES:-1].shape)
	#print(test_y[i].shape)
	test_row = np.concatenate((test_data[i, -NUM_FEATURES:-1], test_y[i]))
	inv_test = scaler.inverse_transform(test_row.reshape(1,-1))
	inv_y = inv_test[:,-1]
	#yhat = invert_scale(scaler, X, predicted_y)
	# invert differencing
	#yhat = inverse_difference(raw_values, yhat, len(test_scaled)+1-i)
	# store forecast
	predictions.append(inv_predicted_y)
	true_values.append(inv_y)
	#expected = raw_values[len(train) + i + 1]
	#print('Day=%d, Predicted=%f, Expected=%f' % (i+1, inv_predicted_y, inv_y))


# report performance
rmse = sqrt(mean_squared_error(true_values, predictions))
mae = mean_absolute_error(true_values, predictions)
print('Test RMSE: %.3f' % rmse)
print('Test MAE: %.3f' % mae)
# line plot of observed vs predicted
#plt.plot(test_data[:, -1])
#plt.plot(predictions)
#plt.show()

# # make a prediction
# predicted_y = lstm_model.predict(test_X, batch_size=test_X.shape[0])
# #print(predicted_y.shape)
# test_X = test_X.reshape((test_X.shape[0], NUM_TIME_STEPS * NUM_FEATURES))
# # invert scaling for forecast
# inv_predicted_y = np.concatenate((test_X[:, -NUM_FEATURES:-1], predicted_y), axis=1)
# inv_predicted_y = scaler.inverse_transform(inv_predicted_y)
# inv_predicted_y = inv_predicted_y[:,-1]
# # invert scaling for actual
# test_y = test_y.reshape((len(test_y), 1))
# inv_y = np.concatenate((test_X[:, -NUM_FEATURES:-1], test_y), axis=1)
# inv_y = scaler.inverse_transform(inv_y)
# inv_y = inv_y[:,-1]
# # calculate RMSE
# rmse = sqrt(mean_squared_error(inv_y, inv_predicted_y))
# print('Test RMSE: %.3f' % rmse)

# #df.close.plot()
# #plt.show()











