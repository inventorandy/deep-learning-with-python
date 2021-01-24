# Import Utils
import os, shutil

# Import Tensorflow
import tensorflow as tf

# Import Keras
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Import Matplotlib
import matplotlib.pyplot as plt

# Get the Current Directory
current_dir = os.path.dirname(os.path.realpath(__file__))

# Directory of Data
data_dir = os.path.join(current_dir, 'data')

# Directory for Original Images
original_data = os.path.join(data_dir, 'original')

# Directory where the small dataset exists
base_dir = os.path.join(data_dir, 'small')
os.mkdir(base_dir)

# Create subdirectories for Training, Validation and Testing
train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)
test_dir = os.path.join(base_dir, 'test')
os.mkdir(test_dir)
validation_dir = os.path.join(base_dir, 'validation')
os.mkdir(validation_dir)

# Now create more subdirectories to split between cats and dogs
train_cats_dir = os.path.join(train_dir, 'cats')
os.mkdir(train_cats_dir)
train_dogs_dir = os.path.join(train_dir, 'dogs')
os.mkdir(train_dogs_dir)
test_cats_dir = os.path.join(test_dir, 'cats')
os.mkdir(test_cats_dir)
test_dogs_dir = os.path.join(test_dir, 'dogs')
os.mkdir(test_dogs_dir)
validation_cats_dir = os.path.join(validation_dir, 'cats')
os.mkdir(validation_cats_dir)
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
os.mkdir(validation_dogs_dir)

# Copy the first 1000 Cat images into the Training Directory
# and then the next 500 into the Validation Directory
# and finally the next 500 into the Test Directory
fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
	src = os.path.join(original_data, fname)
	dst = os.path.join(train_cats_dir, fname)
	shutil.copyfile(src, dst)
fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
	src = os.path.join(original_data, fname)
	dst = os.path.join(validation_cats_dir, fname)
	shutil.copyfile(src, dst)
fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
	src = os.path.join(original_data, fname)
	dst = os.path.join(test_cats_dir, fname)
	shutil.copyfile(src, dst)

# Copy the first 1000 Dog images into the Training Directory
# and then the next 500 into the Validation Directory
# and finally the next 500 into the Test Directory
fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
	src = os.path.join(original_data, fname)
	dst = os.path.join(train_dogs_dir, fname)
	shutil.copyfile(src, dst)
fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
	src = os.path.join(original_data, fname)
	dst = os.path.join(validation_dogs_dir, fname)
	shutil.copyfile(src, dst)
fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
	src = os.path.join(original_data, fname)
	dst = os.path.join(test_dogs_dir, fname)
	shutil.copyfile(src, dst)

# Build the Convolutional Neural Network
# Starting with a feature map size of 150x150 and reducing down to 7x7
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# Print a Summary of the model
model.summary()

# Compile the model using binary crossentropy
model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(learning_rate=1e-4), metrics=['acc'])

# Now we must convert the JPEG files into tensors with RGB values scaled in the range of 0 <= x <= 1
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

## Load and Resize the Images to 150x150
train_generator = train_datagen.flow_from_directory(train_dir, target_size=(150, 150), batch_size=20, class_mode='binary')
validation_generator = test_datagen.flow_from_directory(validation_dir, target_size=(150, 150), batch_size=20, class_mode='binary')

# Fit the model using a batch generator
history = model.fit_generator(train_generator, steps_per_epoch=100, epochs=30, validation_data=validation_generator, validation_steps=50)

# Save the Model
model.save('cats_and_dogs_small_1.h5')

history_dict = history.history
loss = history_dict['loss']
val_loss = history_dict['val_loss']
acc = history_dict['acc']
val_acc = history_dict['val_acc']

epochs = range(1, len(acc) + 1)

# Show Training and Validation Accuracy and Loss
plt.plot(epochs, acc, 'bo', label='Training Accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.savefig('small-dataset-accuracy-1.png')

plt.plot(epochs, loss, 'bo', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig('small-dataset-loss-1.png')

plt.show()

# Build the Convolutional Neural Network
# Starting with a feature map size of 150x150 and reducing down to 7x7
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.5))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# Print a Summary of the model
model.summary()

# Compile the model using binary crossentropy
model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(learning_rate=1e-4), metrics=['acc'])

# Now we must convert the JPEG files into tensors with RGB values scaled in the range of 0 <= x <= 1
train_datagen = ImageDataGenerator(rescale=1./255,
									rotation_range=40,
									width_shift_range=0.2,
									height_shift_range=0.2,
									shear_range=0.2,
									zoom_range=0.2,
									horizontal_flip=True,
									fill_mode='nearest')
test_datagen = ImageDataGenerator(rescale=1./255)

## Load and Resize the Images to 150x150
train_generator = train_datagen.flow_from_directory(train_dir, target_size=(150, 150), batch_size=20, class_mode='binary')
validation_generator = test_datagen.flow_from_directory(validation_dir, target_size=(150, 150), batch_size=20, class_mode='binary')

# Fit the model using a batch generator
history = model.fit_generator(train_generator, steps_per_epoch=100, epochs=30, validation_data=validation_generator, validation_steps=50)

# Save the Model
model.save('cats_and_dogs_small_2.h5')

history_dict = history.history
loss = history_dict['loss']
val_loss = history_dict['val_loss']
acc = history_dict['acc']
val_acc = history_dict['val_acc']

epochs = range(1, len(acc) + 1)

# Show Training and Validation Accuracy and Loss
plt.plot(epochs, acc, 'bo', label='Training Accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.savefig('small-dataset-accuracy-2.png')

plt.plot(epochs, loss, 'bo', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig('small-dataset-loss-2.png')