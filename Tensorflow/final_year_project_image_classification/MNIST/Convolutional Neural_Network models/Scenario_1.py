import numpy as np
import pickle
import os
import tensorflow as tf
import tensorflowvisu
import math
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data as mnist_data

# load data
mnist = mnist_data.read_data_sets("data", one_hot=True, reshape=False, validation_size=0)
# tf.placeholder(dtype, shape=None, name=None) just defines a process
X=tf.placeholder(tf.float32, [None, 28,28,1])

# correct answers
Y_=tf.placeholder(tf.float32, [None, 10])

# weights tensor
W1 = tf.Variable(tf.truncated_normal([5, 5, 1, 4], stddev=0.1))  
B1 = tf.Variable(tf.ones([4])/10)

W3 = tf.Variable(tf.truncated_normal([14 * 14 * 4, 200], stddev=0.1))
B3 = tf.Variable(tf.ones([200])/10)

W4 = tf.Variable(tf.truncated_normal([200, 10], stddev=0.1))
B4 = tf.Variable(tf.ones([10])/10)

stride = 1
#---------------------------------------- Experiment 1 ---------------------------------------------
Y1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='SAME') + B1)
Y2 = tf.layers.max_pooling2d(inputs=Y1, pool_size=[2, 2], strides=2)
Y3 = tf.nn.relu(tf.matmul(tf.reshape(Y2, shape=[-1, 14 * 14 * 4]), W3) + B3)
#---------------------------------------- Experiment 2 ---------------------------------------------
#Y1 = tf.nn.sigmoid(tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='SAME') + B1)
#Y2 = tf.layers.max_pooling2d(inputs=Y1, pool_size=[2, 2], strides=2)
#Y3 = tf.nn.sigmoid(tf.matmul(tf.reshape(Y2, shape=[-1, 14 * 14 * 4]), W3) + B3)
#---------------------------------------- Experiment 3 ---------------------------------------------
#Y1 = tf.nn.softmax(tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='SAME') + B1)
#Y2 = tf.layers.max_pooling2d(inputs=Y1, pool_size=[2, 2], strides=2)
#Y3 = tf.nn.softmax(tf.matmul(tf.reshape(Y2, shape=[-1, 14 * 14 * 4]), W3) + B3)
#---------------------------------------- Experiment 4 ---------------------------------------------
#Y1 = tf.nn.tanh(tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='SAME') + B1)
#Y2 = tf.layers.max_pooling2d(inputs=Y1, pool_size=[2, 2], strides=2)
#Y3 = tf.nn.tanh(tf.matmul(tf.reshape(Y2, shape=[-1, 14 * 14 * 4]), W3) + B3)
#---------------------------------------- Experiment 5 ---------------------------------------------
#Y1 = tf.nn.softplus(tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='SAME') + B1)
#Y2 = tf.layers.max_pooling2d(inputs=Y1, pool_size=[2, 2], strides=2)
#Y3 = tf.nn.softplus(tf.matmul(tf.reshape(Y2, shape=[-1, 14 * 14 * 4]), W3) + B3)
#---------------------------------------- Experiment 6 ---------------------------------------------
#Y1 = tf.nn.elu(tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='SAME') + B1)
#Y2 = tf.layers.max_pooling2d(inputs=Y1, pool_size=[2, 2], strides=2)
#Y3 = tf.nn.elu(tf.matmul(tf.reshape(Y2, shape=[-1, 14 * 14 * 4]), W3) + B3)

Ylogits = tf.matmul(Y3, W4) + B4
Y = tf.nn.softmax(Ylogits)

#loss function
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
cross_entropy = tf.reduce_mean(cross_entropy)*100
#evaluate the performance of the neural network
is_correct=tf.equal(tf.argmax(Y,1), tf.argmax(Y_,1))
accuracy=tf.reduce_mean(tf.cast(is_correct, tf.float32))

allweights = tf.concat([tf.reshape(W1, [-1]), tf.reshape(W3, [-1]), tf.reshape(W4, [-1])], 0)
allbiases  = tf.concat([tf.reshape(B1, [-1]), tf.reshape(B3, [-1]), tf.reshape(B4, [-1])], 0)

I = tensorflowvisu.tf_format_mnist_images(X, Y, Y_)
It = tensorflowvisu.tf_format_mnist_images(X, Y, Y_, 1000, lines=25)
datavis = tensorflowvisu.MnistDataVis()

optimizer=tf.train.GradientDescentOptimizer(0.003)
train_step=optimizer.minimize(cross_entropy)

# init
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

def training_step(i, update_test_data, update_train_data):
    batch_X, batch_Y = mnist.train.next_batch(100)
   
    #train
    if update_train_data:
        a, c, im, w, b = sess.run([accuracy, cross_entropy, I, allweights, allbiases], {X: batch_X, Y_: batch_Y})
        print(str(i) + ": accuracy:" + str(a) + " loss: " + str(c))
        datavis.append_training_curves_data(i, a, c)
        datavis.append_data_histograms(i, w, b)
        datavis.update_image1(im)
       
    #test   
    if update_test_data:
        a, c, im= sess.run([accuracy, cross_entropy, It], {X: mnist.test.images, Y_: mnist.test.labels})
        print(str(i) + ": ********* epoch " + str(i*100//mnist.train.images.shape[0]+1) + " ********* test accuracy:" + str(a) + " test loss: " + str(c))
        datavis.append_test_curves_data(i,a,c)
        datavis.update_image2(im)
    #backpropagation training step
    sess.run(train_step, {X: batch_X, Y_: batch_Y})

datavis.animate(training_step, iterations=10000+1, train_data_update_freq=10, test_data_update_freq=50, more_tests_at_start=True)
print("max test accuracy: " + str(datavis.get_max_test_accuracy())) 