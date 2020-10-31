# Import Tensorflow
import tensorflow as tf

# Import Keras
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras import layers

# Import Numpy
import numpy as np

# Import Matplotlib
import matplotlib.pyplot as plt

# Import the Reuters Dataset
from tensorflow.keras.datasets import boston_housing

# Import the Arguments Parser
import argparse

# Build the Arguments Parser
parser = argparse.ArgumentParser()
parser.add_argument("predict", help="Predict and show the first ten results from the dataset")
args = parser.parse_args()

# Load the House Price Data - each data point has 13 metrics (e.g. crime rate, number of rooms,
# access to highway etc.)
(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

# For each metric, we need to normalise the data in some way to provide numbers around the range
# of zero. For each metric, we subtract the mean and divide by the standard deviation.
mean = train_data.mean(axis = 0)
train_data -= mean
std = train_data.std(axis = 0)
train_data /= std

# Apply same rules to the test data
test_data -= mean
test_data /= std

print(test_targets[0])
print(test_data[0])

# Build the Training Model - we only have a small amount of data so to mitigate overfitting
# we build a small network with two hidden layers. We end with a single unit (no sigmoid
# activation) as we want the algorithm to predict free values (rather than constrain to
# 0 <= {x} <= 1 with sigmoid).
# We return the 'mean absolute error' and 'mean squared error' as metrics on the training.
def build_model():
	model = models.Sequential()
	model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
	model.add(layers.Dense(64, activation='relu'))
	model.add(layers.Dense(1))
	model.compile(optimizer='rmsprop', loss='mse', metrics='mae')
	return model

# Only run this part if we don't have the predict flag written
if not args.predict:
	# Because of the size of our dataset, we use 'k-fold' validation, which splits the data into
	# partitions and instantiating identical models on each partition like so:
	#
	# FOLD 1 == [validation], [training], [training]
	# FOLD 2 == [training], [validation], [training]
	# FOLD 3 == [training], [training], [validation]
	k = 4
	num_val_samples = len(train_data) // k
	num_epochs = 100
	all_scores = []
	all_mae_histories = []

	for i in range(k):
		print("Processing Fold #", i)

		# Prepare the Validation Data from Partition {k}
		val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
		val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]

		# Prepare the Training Data from Partition {k}
		partial_train_data = np.concatenate(
			[train_data[:i * num_val_samples],
			train_data[(i + 1) * num_val_samples:]],
			axis = 0)
		partial_train_targets = np.concatenate([train_targets[:i * num_val_samples], train_targets[(i + 1) * num_val_samples:]], axis = 0)

		# Build the Keras Model
		model = build_model()

		# Train the Model (in silent mode - verbose = 0)
		history = model.fit(partial_train_data, partial_train_targets, epochs=num_epochs, batch_size=1, verbose=1)
		print(history.history.keys())
		mae_history = history.history['mae']
		all_mae_histories.append(mae_history)

		# Get the Mean Absolute Error and Mean Squared Error
		val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
		all_scores.append(val_mae)

	# Calculate the average MAE history
	average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]

	# Plut the Validation MAE
	plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
	plt.xlabel('Epochs')
	plt.ylabel('Validation MAE')
	plt.show()

	# Method to Smooth the Graph Curve
	def smooth_curve(points, factor=0.9):
		smoothed_points = []
		for point in points:
			if smoothed_points:
				previous = smoothed_points[-1]
				smoothed_points.append(previous * factor + point * (1 - factor))
			else:
				smoothed_points.append(point)
		return smoothed_points

	# Cut the First 10 Graph Points
	smoothed_mae_history = smooth_curve(average_mae_history[10:])

	# Plot the New Graph
	plt.plot(range(1, len(smoothed_mae_history) + 1), smoothed_mae_history)
	plt.xlabel('Epochs')
	plt.ylabel('Validation MAE')
	plt.show()

# Get a Fresh Compiled Model and train with 80 epochs, batch size 16
model = build_model()
model.fit(train_data, train_targets, epochs=100, batch_size=16)

# Get the Test Results
test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)
print("Mean Absolute Error: ", test_mae_score)

# Run the Predictions
predictions = model.predict(test_data)
for i in range(10):
	print("Predicted Price: $" + str(predictions[i][0]) + "k Actual Price $" + str(test_targets[i]) + "k")