import numpy as np
import pickle
import os
import tensorflow as tf
import tensorflowvisu
import math
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data as mnist_data

test_accuracy=[]
test_iteration=[]
test_loss=[]
train_accuracy=[]
train_iteration=[]
train_loss=[]
Max_accuracy=0
Min_loss=1000

def draw_plot(test_accuracy,train_accuracy,test_iteration, train_iteration, test_loss, train_loss):
    plt.figure(figsize=(16,7))
    plt.subplot(121) 
    plt.plot(test_iteration,test_accuracy,label="test",color="red",linewidth=1)
    plt.plot(train_iteration,train_accuracy,label="train",color="blue",linewidth=1)
    plt.xlabel("iteration")
    plt.ylabel("accuracy")
    plt.ylim(0,1)
    plt.title("Accuracy")
    plt.legend()
    plt.subplot(122) 
    plt.plot(test_iteration,test_loss,label="test",color="red",linewidth=1)
    plt.plot(train_iteration,train_loss,label="train",color="blue",linewidth=1)
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.ylim(0,100)
    plt.title("Loss")
    plt.legend()
    plt.show()

# load data
mnist = mnist_data.read_data_sets("data", one_hot=True, reshape=False, validation_size=0)
# tf.placeholder(dtype, shape=None, name=None) just defines a process
X=tf.placeholder(tf.float32, [None, 28,28,1])
# variable learning rate
lr = tf.placeholder(tf.float32)
# correct answers
Y_=tf.placeholder(tf.float32, [None, 10])
# the percentage of remaining neurons after dropout
pkeep = tf.placeholder(tf.float32)

# weights tensor
W1 = tf.Variable(tf.truncated_normal([5, 5, 1, 4], stddev=0.1))  
B1 = tf.Variable(tf.ones([4])/10)

W3 = tf.Variable(tf.truncated_normal([4, 4, 4, 12], stddev=0.1))
B3 = tf.Variable(tf.ones([12])/10)

W5 = tf.Variable(tf.truncated_normal([7 * 7 * 12, 200], stddev=0.1))
B5 = tf.Variable(tf.ones([200])/10)

W6 = tf.Variable(tf.truncated_normal([200, 10], stddev=0.1))
B6 = tf.Variable(tf.ones([10])/10)

stride = 1  # output is still 28x28
Y1 = tf.nn.elu(tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='SAME') + B1)
Y2 = tf.layers.max_pooling2d(inputs=Y1, pool_size=[2, 2], strides=2)

stride = 1  # output is 14x14
Y3 = tf.nn.elu(tf.nn.conv2d(Y2, W3, strides=[1, stride, stride, 1], padding='SAME') + B3)
Y4 = tf.layers.max_pooling2d(inputs=Y3, pool_size=[2, 2], strides=2)

Y5 = tf.nn.relu(tf.matmul(tf.reshape(Y4, shape=[-1, 7 * 7 * 12]), W5) + B5)
YY5 = tf.nn.dropout(Y5, pkeep)
Ylogits = tf.matmul(YY5, W6) + B6
Y = tf.nn.softmax(Ylogits)

#loss function
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
cross_entropy = tf.reduce_mean(cross_entropy)*100
#evaluate the performance of the neural network
is_correct=tf.equal(tf.argmax(Y,1), tf.argmax(Y_,1))
accuracy=tf.reduce_mean(tf.cast(is_correct, tf.float32))
# matplotlib visualisation
allweights = tf.concat([tf.reshape(W1, [-1]), tf.reshape(W3, [-1]), tf.reshape(W5, [-1]), tf.reshape(W6, [-1])], 0)
allbiases  = tf.concat([tf.reshape(B1, [-1]), tf.reshape(B3, [-1]), tf.reshape(B5, [-1]), tf.reshape(B6, [-1])], 0)

#I = tensorflowvisu.tf_format_mnist_images(X, Y, Y_)
#It = tensorflowvisu.tf_format_mnist_images(X, Y, Y_, 1000, lines=25)
#datavis = tensorflowvisu.MnistDataVis()

train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

# init
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


def training_step(i, update_test_data, update_train_data):
    batch_X, batch_Y = mnist.train.next_batch(100)
	# learning rate decay
    global Max_accuracy
    global Min_loss
    global test_accuracy
    global test_iteration
    global test_loss
    global train_accuracy
    global train_iteration
    global train_loss
    lrmax = 0.003
    lrmin = 0.0001
    decay_speed = 10000.0 
    learning_rate = lrmin + (lrmax - lrmin) * math.exp(-i/decay_speed)
   
    #train
    if update_train_data:
        a, c, w, b = sess.run([accuracy, cross_entropy, allweights, allbiases], {X: batch_X, Y_: batch_Y, pkeep: 1.0})
        print(str(i) + ": accuracy:" + str(a) + " loss: " + str(c) + " (lr:" + str(learning_rate) + ")")
        #print(str(i) + ": accuracy:" + str(a) + " loss: " + str(c))
        #datavis.append_training_curves_data(i, a, c)
        #datavis.append_data_histograms(i, w, b)
        train_accuracy.append(a)
        train_loss.append(c)
        train_iteration.append(i)
        datavis.update_image1(im)
       
    #test   
    if update_test_data:
        a, c= sess.run([accuracy, cross_entropy], {X: mnist.test.images, Y_: mnist.test.labels, pkeep: 1.0})
        train_accuracy.append(a)
        train_loss.append(c)
        train_iteration.append(i)
        if a>Max_accuracy:
            Max_accuracy=a
        if c<Min_loss:
            Min_loss=c
        print(str(i) + ": ********* epoch " + str(i*100//mnist.train.images.shape[0]+1) + " ********* test accuracy:" + str(a) + " test loss: " + str(c))
        #datavis.append_test_curves_data(i,a,c)
        #datavis.update_image2(im)
    #backpropagation training step
    sess.run(train_step, {X: batch_X, Y_: batch_Y, lr: learning_rate, pkeep: 0.75})
for i in range(_num_iteration+1):
    training_step(i,i%batch_size==0,i%(batch_size//(_num_images_train//_num_images_test))==0)
print("max test accuracy: " + str(Max_accuracy)+" | Min loss: "+str(Min_loss))
#datavis.animate(training_step, iterations=10000+1, train_data_update_freq=10, test_data_update_freq=50, more_tests_at_start=True)
#print("max test accuracy: " + str(datavis.get_max_test_accuracy())) 
