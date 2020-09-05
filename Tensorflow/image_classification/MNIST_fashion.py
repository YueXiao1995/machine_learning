# classification of the clothing images
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)
print(tf.test.is_gpu_available())

fasion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fasion_mnist.load_data()

# map the labels to a classes
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
# look at the size of data
print(train_images.shape)
print(len(train_labels))
print(train_labels)

print(test_images.shape)
print(len(test_labels))
print(test_labels)

# look at a image
print(train_images[0])
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

# scale the value of each pixel to a range of 0 to 1, so that we can feed it into the neural network model
train_images = train_images / 255.0
test_images = test_images / 255.0
print(train_images[0])

# look at the images after the data processing
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

# build the model:
# building the neural network requires two steps:
#   1. configure the layers of the model
#   2. compiling the model

# set up the layers, the basic building block of a neural network is the layer
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)), # transforms the images to a one-dimensional array (28, 28) -> (784)
    keras.layers.Dense(128, activation='relu'), # densely or fully connected neuron layer, this layer has 128 neurons, the activation function is relu
    keras.layers.Dense(10, activation='softmax') # this layer has 10 neurons, the activation function is softmax
])

# before the training we need a few more settings
#   1. loss function: this measures how accurate the model is during training.
#   2. optimizer: this is how the model is updated based on the data it sees and its loss function
#   3. metrics: used to monitor the training and testing steps.

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# train the model
model.fit(train_images, train_labels, epochs=10)

# evaluate the accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

# the result is lower than the training accuracy, this is called overfitting
# Overfitting is when a machine learning model performs worse on new, previously unseen inputs than on the training data

# make predictions
predictions = model.predict(test_images)
print(predictions[0])

# get the label with the highest confidence for the image
print(np.argmax(predictions[0]))
# the real label
print(test_labels[0])

# define two functions to graph the prediction result

def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array, true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'green'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[int(predicted_label)],
                                    100*np.max(predictions_array),
                                    class_names[true_label]),
                                    color=color)

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array, true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('green')

# plot two examples
i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()

i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()

# plot multiple examples
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()

# use the trained model make a prediction about a single image
img = test_images[1]
print(img.shape)
# tf.keras models are optimized to make prediction on a bath, so here need to add another dimension
img = (np.expand_dims(img, 0))
print(img.shape)
# make prediction
predictions_single = model.predict(img)
print(predictions_single)
plot_value_array(1, predictions_single[0], test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)

class_index = int(np.argmax(predictions_single[0]))
print(class_names[class_index])
