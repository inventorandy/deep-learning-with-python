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

# Import the IMDB Dataset
from tensorflow.keras.datasets import imdb

# Load the IMDB Dataset with the top 10,000 Most Frequent Words
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

def decode_review(input_review):
    # Get the Word Index
    word_index = imdb.get_word_index()
    reverse_word_index = dict(
        [(value, key) for (key, value) in word_index.items()]
    )
    decoded_review = ' '.join(
        [reverse_word_index.get(i - 3, '') for i in input_review]
    )
    return decoded_review

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

# Vectorize the Labels
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

# Create the Model (2 layers of relu operations with 16 hidden units each)
# Sigmoid to return a value of between 0 and 1 for probability.
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# Create a Validtion set to monitor accuracy of training
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

# Compile the Model
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

# Begin training the Model with 20 epochs (number of passes)
history = model.fit(partial_x_train, partial_y_train, epochs=10, batch_size=512, validation_data=(x_val, y_val))

# Get the History Values
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']
print("HISTORY KEYS:")
print(history_dict.keys())

# Print a Graph to show Training and Validation Loss
epochs = range(1, len(history_dict['acc']) + 1)
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
results = model.evaluate(x_test, y_test)
predictions = model.predict(x_test)

for i in range(20):
    print(decode_review(test_data[i]))
    print(predictions[i])
    if predictions[i] <= .25:
        print('NEGATIVE')
    if predictions[i] >= .75:
        print('POSITIVE')
    if predictions[i] > .25 and predictions[i] < .75:
        print('UNSURE')
    print('--------------------')