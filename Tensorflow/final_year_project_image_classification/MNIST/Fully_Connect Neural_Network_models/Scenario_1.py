import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data as mnist_data

mnist = mnist_data.read_data_sets("data", one_hot=True, reshape=False, validation_size=0)

X=tf.placeholder(tf.float32, [None, 28,28,1])
# the actual class
Y_=tf.placeholder(tf.float32, [None, 10])
W=tf.Variable(tf.zeros([784, 10]))
b=tf.Variable(tf.zeros([10]))

#prediction
#Y=tf.nn.softmax(tf.matmul(tf.reshape(X, [-1, 784]), W) +b)    #**************experiment 1
#Y=tf.nn.relu(tf.matmul(tf.reshape(X, [-1, 784]), W) +b)       #**************experiment 2
#Y=tf.nn.sigmoid(tf.matmul(tf.reshape(X, [-1, 784]), W) +b)    #**************experiment 3
#Y=tf.nn.tanh(tf.matmul(tf.reshape(X, [-1, 784]), W) +b)       #**************experiment 4
#Y=tf.nn.softplus(tf.matmul(tf.reshape(X, [-1, 784]), W) +b)   #**************experiment 5
#Y=tf.nn.elu(tf.matmul(tf.reshape(X, [-1, 784]), W) +b)        #**************experiment 6


allweights = tf.reshape(W, [-1])
allbiases = tf.reshape(b, [-1])


#loss function
cross_entropy=-tf.reduce_mean(Y_* tf.log(Y))*100

is_correct=tf.equal(tf.argmax(Y,1), tf.argmax(Y_,1))
accuracy=tf.reduce_mean(tf.cast(is_correct, tf.float32))
#gradient descent
optimizer=tf.train.GradientDescentOptimizer(0.003)
train_step=optimizer.minimize(cross_entropy)



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
    #plt.savefig("tanh.jpg")
    plt.savefig("softplus.jpg")

init=tf.initialize_all_variables()
sess=tf.Session()
sess.run(init)
sess.graph.finalize()



def training_step(i, update_test_data, update_train_data):
    batch_X, batch_Y = mnist.train.next_batch(100)
    global Max_accuracy
    global Min_loss
    global test_accuracy
    global test_iteration
    global test_loss
    global train_accuracy
    global train_iteration
    global train_loss
    if update_train_data:
        a, c, w, b = sess.run([accuracy, cross_entropy, allweights, allbiases], feed_dict={X: batch_X, Y_: batch_Y})
        train_accuracy.append(a)
        train_loss.append(c)
        train_iteration.append(i)
        print(str(i) + ": accuracy:" + str(a) + " loss: " + str(c))

    if update_test_data:
        a, c= sess.run([accuracy, cross_entropy], feed_dict={X: mnist.test.images, Y_: mnist.test.labels})
        test_accuracy.append(a)
        test_loss.append(c)
        test_iteration.append(i)
        if a>Max_accuracy:
            Max_accuracy=a
        if c<Min_loss:
            Min_loss=c
        print(str(i) + ": ********* epoch " + str(i*100//mnist.train.images.shape[0]+1) + " ********* test accuracy:" + str(a) + " test loss: " + str(c))

    sess.run(train_step, feed_dict={X: batch_X, Y_: batch_Y})

for i in range(10001):
    training_step(i,i%100==0,i%20==0)
print("max test accuracy: " + str(Max_accuracy)+" | Min loss: "+str(Min_loss))
draw_plot(test_accuracy,train_accuracy,test_iteration, train_iteration, test_loss, train_loss)
sess.close()
