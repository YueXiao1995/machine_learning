import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
from tensorflow.examples.tutorials.mnist import input_data as mnist_data

#load data
mnist = mnist_data.read_data_sets("data", one_hot=True, reshape=False, validation_size=0)
#tf.placeholder(dtype, shape=None, name=None) just defines a process
X=tf.placeholder(tf.float32, [None, 28,28,1])
# variable learning rate
lr = tf.placeholder(tf.float32)
pkeep = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.truncated_normal([28*28, 200] ,stddev=0.1))
B1 = tf.Variable(tf.ones([200])/10)
W2 = tf.Variable(tf.truncated_normal([200, 100] ,stddev=0.1))
B2 = tf.Variable(tf.ones([100])/10)
W3 = tf.Variable(tf.truncated_normal([100,60] ,stddev=0.1))
B3 = tf.Variable(tf.ones([60])/10)
W4 = tf.Variable(tf.truncated_normal([60,30] ,stddev=0.1))
B4 = tf.Variable(tf.ones([30])/10)
W5 = tf.Variable(tf.truncated_normal([30, 10], stddev=0.1))
B5 = tf.Variable(tf.zeros([10]))

XX = tf.reshape(X, [-1, 28*28])
Y1 = tf.nn.relu(tf.matmul(XX, W1) + B1)
Y2 = tf.nn.relu(tf.matmul(Y1, W2) + B2)
Y3 = tf.nn.relu(tf.matmul(Y2, W3) + B3)
Y4 = tf.nn.relu(tf.matmul(Y3, W4) + B4)
YY4 = tf.nn.dropout(Y4, pkeep)
Ylogits = tf.matmul(YY4, W5) + B5
Y = tf.nn.softmax(Ylogits)
Y_=tf.placeholder(tf.float32, [None, 10])

allweights = tf.concat([tf.reshape(W1, [-1]),tf.reshape(W2, [-1]), tf.reshape(W3, [-1]), tf.reshape(W4, [-1]), tf.reshape(W5, [-1])], 0)
allbiases  = tf.concat([tf.reshape(B1, [-1]),tf.reshape(B2, [-1]), tf.reshape(B3, [-1]), tf.reshape(B4, [-1]), tf.reshape(B5, [-1])], 0)

#loss function
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
cross_entropy = tf.reduce_mean(cross_entropy)*100
#evaluate the performance of the neural network
is_correct=tf.equal(tf.argmax(Y,1), tf.argmax(Y_,1))
accuracy=tf.reduce_mean(tf.cast(is_correct, tf.float32))

#gradient descent, learning rate 0.003
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

Max_accuracy=0
Min_loss=10000

test_accuracy=[]
test_iteration=[]
test_loss=[]
train_accuracy=[]
train_iteration=[]
train_loss=[]

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
    #plt.savefig("softsoft.jpg")
    #plt.savefig("softrelu.jpg")
    #plt.savefig("softelu.jpg")
    plt.savefig("softsigmoid.jpg")

init=tf.initialize_all_variables()
sess=tf.Session()
sess.run(init)
sess.graph.finalize()

def training_step(i, update_test_data, update_train_data):
    batch_X, batch_Y = mnist.train.next_batch(100)
    
   # learning rate decay
    lrmax = 0.003
    lrmin = 0.0001
    decay_speed = 10000.0 
    learning_rate = lrmin + (lrmax - lrmin) * math.exp(-i/decay_speed)
    global Max_accuracy
    global Min_loss
    global test_accuracy
    global test_iteration
    global test_loss
    global train_accuracy
    global train_iteration
    global train_loss
    #train
    if update_train_data:
        a, c, w, b = sess.run([accuracy, cross_entropy, allweights, allbiases], feed_dict={X: batch_X, Y_: batch_Y, pkeep: 1.0})
        train_accuracy.append(a)
        train_loss.append(c)
        train_iteration.append(i)
        print(str(i) + ": accuracy:" + str(a) + " loss: " + str(c))

    if update_test_data:
        a, c= sess.run([accuracy, cross_entropy], feed_dict={X: mnist.test.images, Y_: mnist.test.labels, pkeep: 1.0})
        test_accuracy.append(a)
        test_loss.append(c)
        test_iteration.append(i)
        if a>Max_accuracy:
            Max_accuracy=a
        if c<Min_loss:
            Min_loss=c
        print(str(i) + ": ********* epoch " + str(i*100//mnist.train.images.shape[0]+1) + " ********* test accuracy:" + str(a) + " test loss: " + str(c))

    sess.run(train_step, feed_dict={X: batch_X, Y_: batch_Y, lr: learning_rate, pkeep: 0.75})

for i in range(10001):
    training_step(i,i%100==0,i%20==0)
print("max test accuracy: " + str(Max_accuracy)+" | Min loss: "+str(Min_loss))

draw_plot(test_accuracy,train_accuracy,test_iteration, train_iteration, test_loss, train_loss)

sess.close()


