# classification of hand written digits images
# this program uses tf.keras, a high-level API to build and train models in TensorFlow.
import tensorflow as tf

# remove the warning message
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# load data
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# convert the samples from integers to floating point numbers
x_train, x_test = x_train / 255.0, x_test / 255.0

# build a model
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# train the model
model.fit(x_train, y_train, epochs=5)

# evaluation the model
model.evaluate(x_test,  y_test, verbose=2)
