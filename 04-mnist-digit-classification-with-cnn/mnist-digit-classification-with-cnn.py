# Import Tensorflow
import tensorflow as tf

# Import Keras
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras import layers

# Import the MNIST Dataset
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Import Matplotlib
import numpy as np
import matplotlib.pyplot as plt

# Create the Convolutional Neural Network (convnet)
# This takes input tensors of 3D (height, width, channels)
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# Now flatten the output into 1D tensors
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax')) # Classify with 10 outputs (digits 0-9)

# Print a Summary of the model
model.summary()

# Load the MNIST Dataset into Training and Test Sets
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Convert the Training Images to a group of 60,000 images of 28x28 pixels
train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255

# Convert the Test Images to a group of 10,000 images of 28x28 pixels
test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255

# Categorise the Labels
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Compile the Model and Train
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5, batch_size=64)

# Run the Model against the Test Data
test_loss, test_acc = model.evaluate(test_images, test_labels)

# Print the Results
print('Test Accuracy ' + str(test_acc) + '%')

# Now generate some predictions
probability_model = keras.Sequential([model, layers.Softmax()])
predictions = probability_model.predict(test_images)

# Method for Plotting the Image
def plot_image(i, predictions_array, true_labels, imgs):
	true_label, img = true_labels[i], imgs[i]
	plt.grid(False)
	plt.xticks([])
	plt.yticks([])

	plt.imshow(img)

	predicted_label = np.argmax(predictions_array)
	true_label = np.argmax(true_label)
	if predicted_label == true_label:
		color = 'blue'
	else:
		color = 'red'
	
	plt.xlabel("Pred {} (Actual {})".format(predicted_label,
									  true_label),
									  color=color)

# Plot the first 25 predictions
plt.figure(figsize=(10,10))
for i in range(25):
	plt.subplot(5,5,i+1)
	plot_image(i, predictions[i], test_labels, test_images)
plt.savefig("predictions.png")