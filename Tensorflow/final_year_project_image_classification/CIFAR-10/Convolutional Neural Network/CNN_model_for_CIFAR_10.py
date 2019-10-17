import tensorflow as tf
import math
import numpy as np
import pickle
import os
#import matplotlib.pyplot as plt
#Select	the GPU. GPUs available in theengine2: 0,1,2,3

#os.environ["CUDA_VISIBLE_DEVICES"]="3"

#If we select more than 1 GPU and we don't want to allocate all the memory:
config = tf.ConfigProto()
config.gpu_options.allow_growth=True

print("Tensorflow version " + tf.__version__)
tf.set_random_seed(0)

data_path = "/home/yue/data/"
img_size = 32
num_channels = 3
img_size_flat = img_size * img_size * num_channels
num_classes = 10
_num_files_train = 5
_images_per_file = 10000
_num_images_train = _num_files_train * _images_per_file
_num_images_test=10000
epoch=50
batch_size=100
_num_iteration=(_num_images_train//batch_size)*epoch
test_accuracy=[]
test_iteration=[]
test_loss=[]
train_accuracy=[]
train_iteration=[]
train_loss=[]

def _get_file_path(filename=""):
    return os.path.join(data_path, "cifar-10-batches-py/", filename)

def _unpickle(filename):
    file_path = _get_file_path(filename)
    
    print("Loading data: " + file_path)

    with open(file_path, mode='rb') as file:
        data = pickle.load(file, encoding='bytes')
        
    return data
def _convert_images(raw):
    raw_float = np.array(raw, dtype=float) / 255.0
    images = raw_float.reshape([-1, num_channels, img_size, img_size])

    # Reorder the indices of the array.
    images = images.transpose([0, 2, 3, 1])
    #images=_grayscale(images)
    return images
def _load_data(filename):
    # Load the pickled data-file.
    data = _unpickle(filename)

    # Get the raw images.
    raw_images = data[b'data']

    # Get the class-numbers for each image. Convert to numpy-array.
    labels = np.array(data[b'labels'])
    images = _convert_images(raw_images)
    return images, labels

def load_training_data():
  
    # Pre-allocate the arrays for the images and class-numbers for efficiency.
    images = np.zeros(shape=[_num_images_train, img_size, img_size, num_channels], dtype=float)
    cls = np.zeros(shape=[_num_images_train], dtype=int)

    # Begin-index for the current batch.
    begin = 0

    # For each data-file.
    for i in range(_num_files_train):
        # Load the images and class-numbers from the data-file.
        images_batch, cls_batch = _load_data(filename="data_batch_" + str(i + 1))

        # Number of images in this batch.
        num_images = len(images_batch)

        # End-index for the current batch.
        end = begin + num_images

        # Store the images into the array.
        images[begin:end, :] = images_batch

        # Store the class-numbers into the array.
        cls[begin:end] = cls_batch
        #cls.shape()
        # The begin-index for the next batch is the current end-index.
        begin = end
        #onehot_labels=tf.one_hot(indices=tf.cast(cls, tf.int32), depth=10)
    #images=np.array(images)
    return images, cls

def load_test_data():

    images, cls = _load_data(filename="test_batch")
    #onehot_labels=tf.one_hot(indices=tf.cast(cls, tf.int32), depth=10)
    return images, cls
	
train_images,train_labels=load_training_data()
test_images,test_labels=load_test_data()

def next_train_batch_images(begin, batch_size):
    end=begin+batch_size
    tr_images =train_images[begin:end,:]
    return tr_images

def next_train_batch_labels(begin, batch_size):
    end=begin+batch_size
    tr_labels =train_labels[begin:end]
    #tr_labels.shape()
    return tr_labels

def next_test_batch_images(begin, batch_size):
    end=begin+batch_size
    te_images = test_images[begin:end,:]
    return te_images

def next_test_batch_labels(begin, batch_size):
    end=begin+batch_size
    te_labels = test_labels[begin:end]
    return te_labels

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
# the input
X = tf.placeholder(tf.float32, [None, 32, 32, 3])
# The out put of neural network
Y_ = tf.placeholder(tf.int32, [batch_size])
one_hot=tf.one_hot(indices=Y_, depth=10)
# the percentage of remaining neurons after dropout
pkeep = tf.placeholder(tf.float32)

#-----------------------------------------------------------------------------------------------------------------------------------
W1 = tf.Variable(tf.truncated_normal([5, 5, 3, 6], stddev=0.1)) 
B1 = tf.Variable(tf.ones([6])/10)
W2 = tf.Variable(tf.truncated_normal([5, 5, 6, 12], stddev=0.1))  
B2 = tf.Variable(tf.ones([12])/10)
W3 = tf.Variable(tf.truncated_normal([5, 5, 12, 24], stddev=0.1))  
B3 = tf.Variable(tf.ones([24])/10)

W5 = tf.Variable(tf.truncated_normal([5, 5, 24, 36], stddev=0.1))  
B5 = tf.Variable(tf.ones([36])/10)
W6 = tf.Variable(tf.truncated_normal([5, 5, 36, 48], stddev=0.1))  
B6 = tf.Variable(tf.ones([48])/10)

W8 = tf.Variable(tf.truncated_normal([5, 5, 48, 72], stddev=0.1))
B8 = tf.Variable(tf.ones([72])/10)
W9 = tf.Variable(tf.truncated_normal([5, 5, 72, 96], stddev=0.1)) 
B9 = tf.Variable(tf.ones([96])/10)

W11 = tf.Variable(tf.truncated_normal([4 * 4 * 96, 200], stddev=0.1))
B11 = tf.Variable(tf.ones([200])/10)
W12 = tf.Variable(tf.truncated_normal([200, 10], stddev=0.1))
B12 = tf.Variable(tf.ones([10])/10)
lr = tf.placeholder(tf.float32)

stride = 1
Y1 = tf.nn.elu(tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='SAME') + B1)# convolutional layer1, output is 32x32
YY1 = tf.nn.dropout(Y1, pkeep)

stride = 1
Y2 = tf.nn.elu(tf.nn.conv2d(YY1, W2, strides=[1, stride, stride, 1], padding='SAME') + B2)# convolutional layer2, output is 32x32
YY2 = tf.nn.dropout(Y2, pkeep)

stride = 1
Y3 = tf.nn.elu(tf.nn.conv2d(YY2, W3, strides=[1, stride, stride, 1], padding='SAME') + B3)# convolutional layer3, output is 32x32
YY3 = tf.nn.dropout(Y3, pkeep) 
Y4 = tf.layers.max_pooling2d(inputs=YY3, pool_size=[2, 2], strides=2)# pooling layer1, out put is 16x16
YY4 = tf.nn.lrn(Y4)

stride = 1
Y5 = tf.nn.elu(tf.nn.conv2d(YY4, W5, strides=[1, stride, stride, 1], padding='SAME') + B5)# convolutional layer4
YY5 = tf.nn.dropout(Y5, pkeep)
stride = 1
Y6 = tf.nn.elu(tf.nn.conv2d(YY5, W6, strides=[1, stride, stride, 1], padding='SAME') + B6)# convolutional layer5
YY6 = tf.nn.dropout(Y6, pkeep)
Y7 = tf.layers.max_pooling2d(inputs=YY6, pool_size=[2, 2], strides=2)# pooling layer2, out put is 8x8
YY7 = tf.nn.lrn(Y7)

stride = 1
Y8 = tf.nn.elu(tf.nn.conv2d(YY7, W8, strides=[1, stride, stride, 1], padding='SAME') + B8)# convolutional layer6
YY8 = tf.nn.dropout(Y8, pkeep)
stride = 1
Y9 = tf.nn.elu(tf.nn.conv2d(YY8, W9, strides=[1, stride, stride, 1], padding='SAME') + B9)# convolutional layer7
YY9 = tf.nn.dropout(Y9, pkeep)
Y10= tf.layers.max_pooling2d(inputs=YY9, pool_size=[2, 2], strides=2)# pooling layer3, out put is 4x4
YY10 = tf.nn.lrn(Y10)

# dense layer
YY = tf.reshape(YY10, shape=[-1, 4 * 4 * 96])
Y11 = tf.nn.elu(tf.matmul(YY, W11) + B11)
YY11 = tf.nn.dropout(Y11, pkeep)
# variable learning rate

# logits layer
Ylogits = tf.matmul(YY11, W12) + B12
Y = tf.nn.softmax(Ylogits)

# matplotlib visualisation
allweights = tf.concat([tf.reshape(W1, [-1]), tf.reshape(W2, [-1]), tf.reshape(W3, [-1]), tf.reshape(W5, [-1]), tf.reshape(W6, [-1]), tf.reshape(W8, [-1]), tf.reshape(W9, [-1]), tf.reshape(W11, [-1]), tf.reshape(W12, [-1])], 0)
allbiases  = tf.concat([tf.reshape(B1, [-1]), tf.reshape(B2, [-1]), tf.reshape(B3, [-1]), tf.reshape(B5, [-1]), tf.reshape(B6, [-1]), tf.reshape(B8, [-1]), tf.reshape(B9, [-1]), tf.reshape(B11, [-1]), tf.reshape(B12, [-1])], 0)
#-----------------------------------------------------------------------------------------------------------------------------------

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=one_hot)
cross_entropy = tf.reduce_mean(cross_entropy)*100

# accuracy of the trained model, between 0 (worst) and 1 (best)
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

# init
init = tf.global_variables_initializer()
sess = tf.Session(config=config)
#sess = tf.Session()
sess.run(init)
sess.graph.finalize() 

Max_accuracy=0
Min_loss=1000


def training_step(i, update_test_data, update_train_data):
    # training on batches of 100 images with 100 labels
    global batch_size
    global Max_accuracy
    global Min_loss
    global test_accuracy
    global test_iteration
    global test_loss
    global train_accuracy
    global train_iteration
    global train_loss

    index=i%(_num_images_train//batch_size)
    # learning rate decay
    max_learning_rate = 0.003
    min_learning_rate = 0.0001
    decay_speed = 15000.0
    learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i/decay_speed)
    
    # compute training values for visualisation
    if update_train_data:
        a, c, w, b = sess.run([accuracy, cross_entropy, allweights, allbiases], feed_dict={X: next_train_batch_images(index*batch_size,batch_size), Y_:next_train_batch_labels(index*batch_size,batch_size), pkeep: 1.0})
        train_accuracy.append(a)
        train_loss.append(c)
        train_iteration.append(i)
        print(str(i) + ": accuracy:" + str(a) + " loss: " + str(c) + " (lr:" + str(learning_rate) + ")")

    # compute test values for visualisation
    if update_test_data:
        a, c = sess.run([accuracy, cross_entropy], feed_dict={X: next_test_batch_images((index%batch_size)*batch_size,batch_size),Y_:next_test_batch_labels((index%batch_size)*batch_size,batch_size), pkeep: 1.0})
        test_accuracy.append(a)
        test_loss.append(c)
        test_iteration.append(i)
        if a>Max_accuracy:
            Max_accuracy=a
        if c<Min_loss:
            Min_loss=c
        print(str(i) + ": ********* epoch " + str((i//(_num_images_train//batch_size))+1) + " ********* test accuracy:" + str(a) + " test loss: " + str(c))

    # the backpropagation training step
    sess.run(train_step, feed_dict={X: next_train_batch_images(index*batch_size,batch_size), Y_:next_train_batch_labels(index*batch_size,batch_size),lr: learning_rate, pkeep: 0.75})

for i in range(_num_iteration+1):
    training_step(i,i%batch_size==0,i%(batch_size//(_num_images_train//_num_images_test))==0)
print("max test accuracy: " + str(Max_accuracy)+" | Min loss: "+str(Min_loss))
# show all of the results record in training process
print("=====================================test_accuracy=======================================")
print(test_accuracy)
print("=======================================test_loss=========================================")
print(test_loss)
print("===================================train_accuracy=========================================")
print(train_accuracy)
print("======================================train_loss=========================================")
print(train_loss)
#draw_plot(test_accuracy,train_accuracy,test_iteration, train_iteration, test_loss, train_loss)
sess.close()
