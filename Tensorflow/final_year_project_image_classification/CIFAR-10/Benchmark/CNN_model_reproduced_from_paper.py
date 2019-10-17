import tensorflow as tf
import math
import numpy as np
import pickle
import os
#Select	the GPU. GPUs available in theengine2: 0,1,2,3
os.environ["CUDA_VISIBLE_DEVICES"]="2"

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
epoch=30
batch_size=100
_num_iteration=(_num_images_train//batch_size)*epoch
test_accuracy=[]
test_iteration=[]
test_loss=[]
train_accuracy=[]
train_iteration=[]
train_loss=[]

def _grayscale(a):
    return a.reshape(a.shape[0],32,32,3).mean(3)

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

        # The begin-index for the next batch is the current end-index.
        begin = end
        #onehot_labels=tf.one_hot(indices=tf.cast(cls, tf.int32), depth=10)
    return images, cls


def load_test_data():

    images, cls = _load_data(filename="test_batch")
    #onehot_labels=tf.one_hot(indices=tf.cast(cls, tf.int32), depth=10)
    return images, cls
	
train_images,train_labels=load_training_data()
test_images,test_labels=load_test_data()

X = tf.placeholder(tf.float32, [None, 32, 32, 3])
#X1 = tf.image.resize_images(X, [227,227],method=tf.image.ResizeMethod.BICUBIC)
# correct answers will go here
#Y_ = tf.placeholder(tf.float32, [None, 10])
Y_ = tf.placeholder(tf.int32, [100])
one_hot=tf.one_hot(indices=Y_, depth=10)
#===================================================================================================================================
"""weight_decay = tf.constant(0.0005, dtype=tf.float32) # your weight decay rate, must be a scalar tensor.
W1 = tf.get_variable(name='weight', shape=[11, 11, 3, 96], regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
#W1 = tf.Variable(tf.truncated_normal([11, 11, 3, 96], stddev=0.1))
B1 = tf.Variable(tf.ones([96])/10)
W4 = tf.get_variable(name='weight4', shape=[5, 5, 96, 256], regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
#W4 = tf.Variable(tf.truncated_normal([5, 5, 96, 256], stddev=0.1))
B4 = tf.Variable(tf.ones([256])/10)
W7 = tf.get_variable(name='weight7', shape=[3, 3, 256, 384], regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
#W7 = tf.Variable(tf.truncated_normal([3, 3, 256, 384], stddev=0.1))
B7 = tf.Variable(tf.ones([384])/10)
W8 = tf.get_variable(name='weight8', shape=[3, 3, 384, 384], regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
#W8 = tf.Variable(tf.truncated_normal([3, 3, 384, 384], stddev=0.1))
B8 = tf.Variable(tf.ones([384])/10)
W9 = tf.get_variable(name='weight9', shape=[3, 3, 384, 256], regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
#W9 = tf.Variable(tf.truncated_normal([3, 3, 384, 256], stddev=0.1))
B9 = tf.Variable(tf.ones([256])/10)
W11 = tf.get_variable(name='weight11', shape=[6*6*256, 4096], regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
#W11 = tf.Variable(tf.truncated_normal([6*6*256, 4096], stddev=0.1))
B11 = tf.Variable(tf.ones([4096])/10)
W12 = tf.get_variable(name='weight12', shape=[1*1*4096, 4096], regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
#W12 = tf.Variable(tf.truncated_normal([1*1*4096, 4096], stddev=0.1))
B12 = tf.Variable(tf.ones([4096])/10)
W13 = tf.get_variable(name='weight13', shape=[4096, 10], regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
#W13 = tf.Variable(tf.truncated_normal([4096, 10], stddev=0.1))
B13 = tf.Variable(tf.ones([10])/10)
lr = tf.placeholder(tf.float32)
pkeep = tf.placeholder(tf.float32)"""
#======================================================================================================================================
weight_decay = tf.constant(0.0005, dtype=tf.float32)
W1 = tf.get_variable(name='weight', shape=[5, 5, 3, 12], regularizer=tf.contrib.layers.l2_regularizer(weight_decay))

B1 = tf.Variable(tf.ones([12])/10)
W4 = tf.get_variable(name='weight4', shape=[3, 3, 12, 20], regularizer=tf.contrib.layers.l2_regularizer(weight_decay))

B4 = tf.Variable(tf.ones([20])/10)
W7 = tf.get_variable(name='weight7', shape=[3, 3, 20, 32], regularizer=tf.contrib.layers.l2_regularizer(weight_decay))

B7 = tf.Variable(tf.ones([32])/10)
W8 = tf.get_variable(name='weight8', shape=[3, 3, 32, 32], regularizer=tf.contrib.layers.l2_regularizer(weight_decay))

B8 = tf.Variable(tf.ones([32])/10)
W9 = tf.get_variable(name='weight9', shape=[3, 3, 32, 20], regularizer=tf.contrib.layers.l2_regularizer(weight_decay))

B9 = tf.Variable(tf.ones([20])/10)
W11 = tf.get_variable(name='weight11', shape=[4*4*20, 64], regularizer=tf.contrib.layers.l2_regularizer(weight_decay))

B11 = tf.Variable(tf.ones([64])/10)
W12 = tf.get_variable(name='weight12', shape=[64, 64], regularizer=tf.contrib.layers.l2_regularizer(weight_decay))

B12 = tf.Variable(tf.ones([64])/10)
W13 = tf.get_variable(name='weight13', shape=[64, 10], regularizer=tf.contrib.layers.l2_regularizer(weight_decay))

B13 = tf.Variable(tf.ones([10])/10)
lr = tf.placeholder(tf.float32)
pkeep = tf.placeholder(tf.float32)
stride = 1  

"""stride=4"""
Y1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='SAME') + B1)

Y2 = tf.layers.max_pooling2d(inputs=Y1, pool_size=[2, 2], strides=2)
"""Y2 = tf.layers.max_pooling2d(inputs=Y1, pool_size=[3, 3], strides=2)"""
Y3 = tf.nn.lrn(Y2)

stride = 1
Y4 = tf.nn.relu(tf.nn.conv2d(Y3, W4, strides=[1, stride, stride, 1], padding='SAME') + B4)

Y5 = tf.layers.max_pooling2d(inputs=Y4, pool_size=[2, 2], strides=2)
"""Y5 = tf.layers.max_pooling2d(inputs=Y1, pool_size=[3, 3], strides=2)"""
Y6 = tf.nn.lrn(Y5)

stride = 1
Y7 = tf.nn.relu(tf.nn.conv2d(Y6, W7, strides=[1, stride, stride, 1], padding='SAME') + B7)

stride = 1
Y8 = tf.nn.relu(tf.nn.conv2d(Y7, W8, strides=[1, stride, stride, 1], padding='SAME') + B8)

stride = 1
Y9 = tf.nn.relu(tf.nn.conv2d(Y8, W9, strides=[1, stride, stride, 1], padding='SAME') + B9)

Y10 = tf.layers.max_pooling2d(inputs=Y9, pool_size=[2, 2], strides=2)
"""Y10 = tf.layers.max_pooling2d(inputs=Y1, pool_size=[3, 3], strides=2)"""

# dense layer
Y11 = tf.nn.relu(tf.matmul(tf.reshape(Y10, shape=[-1, 4 * 4 * 20]), W11) + B11)
"""Y11 = tf.nn.relu(tf.matmul(tf.reshape(Y10, shape=[-1, 6*6*256]), W11) + B11)"""
YY11 = tf.nn.dropout(Y11, pkeep)

Y12 = tf.nn.relu(tf.matmul(tf.reshape(YY11, shape=[-1, 64]), W12) + B12)
"""Y12 = tf.nn.relu(tf.matmul(tf.reshape(YY11, shape=[-1, 4096]), W12) + B12)"""
YY12 = tf.nn.dropout(Y12, pkeep)

Ylogits = tf.matmul(YY12, W13) + B13
Y = tf.nn.softmax(Ylogits)

allweights = tf.concat([tf.reshape(W1, [-1]), tf.reshape(W4, [-1]), tf.reshape(W7, [-1]), tf.reshape(W8, [-1]), tf.reshape(W9, [-1]), tf.reshape(W11, [-1]), tf.reshape(W12, [-1]), tf.reshape(W13, [-1])], 0)
allbiases  = tf.concat([tf.reshape(B1, [-1]), tf.reshape(B4, [-1]), tf.reshape(B7, [-1]), tf.reshape(B8, [-1]), tf.reshape(B9, [-1]), tf.reshape(B11, [-1]), tf.reshape(B12, [-1]), tf.reshape(B13, [-1])], 0)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=one_hot)
cross_entropy = tf.reduce_mean(cross_entropy)*100

correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)


def next_train_batch_images(begin, batch_size):
    end=begin+batch_size
    tr_images = train_images[begin:end,:]
    #tr_images = tf.image.resize_images(tr_images, [227,227])
    return tr_images
    
def next_train_batch_labels(begin, batch_size):
    end=begin+batch_size
    tr_labels = train_labels[begin:end]
    return tr_labels

def next_test_batch_images(begin, batch_size):
    end=begin+batch_size
    te_images = test_images[begin:end,:]
    #te_images = tf.image.resize_images(te_images, [227,227])
    return te_images
    
def next_test_batch_labels(begin, batch_size):
    end=begin+batch_size
    te_labels = test_labels[begin:end]
    return te_labels

# init
init = tf.global_variables_initializer()
#sess = tf.Session(config=config)
sess = tf.Session()
sess.run(init)
sess.graph.finalize() 

Max_accuracy=0
Min_loss=1000

# You can call this function in a loop to train the model, 100 images at a time
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
