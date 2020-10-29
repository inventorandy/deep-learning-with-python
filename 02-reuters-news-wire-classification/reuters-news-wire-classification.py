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
from tensorflow.keras.datasets import reuters

# Import the NP Utils functions
from tensorflow.keras.utils import to_categorical

# Load the Reuters News Stories
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

def decode_newswire(input_newswire):
    # Get the Word Index
    word_index = reuters.get_word_index()
    reverse_word_index = dict(
        [(value, key) for (key, value) in word_index.items()]
    )
    decoded_newswire = ' '.join(
        [reverse_word_index.get(i - 3, '') for i in input_newswire]
    )
    return decoded_newswire

# Method for Vectorizing the Word Sequences (creates an all-zero matrix of shape (len(sequences), dimension))
def vectorize_sequences(sequences, dimension=10000):
    # Create an all-zero matrix
    results = np.zeros((len(sequences), dimension))

    # Set specific indices to 1
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

# Vectorize the Training and Test Data
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

# Embed Categories as an all-zero vector with the index of the category set to one
one_hot_train_labels = to_categorical(train_labels)
one_hot_test_labels = to_categorical(test_labels)

# Create the model - we're using much bigger intermediate layers here as we're classifying between 46
# different classes (rather than two in the IMDB example). Additionally, we use 'softmax' for our last
# layer which gives us the probability distribution over each of the classes.
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

# Compile the model using 'categorical_crossentropy', which measures the distance between the probability
# distribution output by the network and the true distribution of the labels.
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics='acc')

# Create a Validtion set to monitor accuracy of training
x_val = x_train[:1000]
partial_x_train = x_train[1000:]
y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

# Train the model with 10 epochs
history = model.fit(partial_x_train, partial_y_train, epochs=10, batch_size=512, validation_data=(x_val, y_val))

# Get the History Values
history_dict = history.history
print(history_dict)
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']
print("HISTORY KEYS:")
print(history_dict.keys())

# Print a Graph to show Training and Validation Loss
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, 'bo', label='Training Loss') # Print training loss as 'bo' (blue dot)
plt.plot(epochs, val_loss_values, 'b', label='Validation Loss') # Print validation loss as 'b' (blue line)
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Print a Graph to show Training and Validation Accuracy
plt.clf()
plt.plot(epochs, acc_values, 'bo', label='Training Accuracy')
plt.plot(epochs, val_acc_values, 'b', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Now lets run predictions on the test data
results = model.evaluate(x_test, one_hot_test_labels)
predictions = model.predict(x_test)

# List of News Topics ($ref: https://github.com/SteffenBauer/KerasTools/tree/master/Reuters_Analysis)
topics = ['cocoa','grain','veg-oil','earn','acq','wheat','copper','housing','money-supply',
 'coffee','sugar','trade','reserves','ship','cotton','carcass','crude','nat-gas',
 'cpi','money-fx','interest','gnp','meal-feed','alum','oilseed','gold','tin',
 'strategic-metal','livestock','retail','ipi','iron-steel','rubber','heat','jobs',
 'lei','bop','zinc','orange','pet-chem','dlr','gas','silver','wpi','hog','lead']

# Method to return the highest probability
def highest_probability(prediction):
	highest_probability = 0
	highest_probability_index = 0
	for i in range(len(prediction)):
		if prediction[i] > highest_probability:
			highest_probability = prediction[i]
			highest_probability_index = i
	return highest_probability_index

for i in range(10):
	prob = highest_probability(predictions[i])
	print("Article " + str(i) + " Topic: " + topics[prob] + " (" + str(predictions[i][prob] * 100) + "%)")
	print(decode_newswire(test_data[i]))
	print("--------------------")