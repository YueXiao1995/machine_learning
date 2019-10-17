import tensorflow as tf
import math
import numpy as np
import pickle
import os

import matplotlib.pyplot as plt
from PIL import Image
#Select	the GPU. GPUs available in theengine2: 0,1,2,3
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

#If we select more than 1 GPU and we don't want to allocate all the memory:
config = tf.ConfigProto()
config.gpu_options.allow_growth=True


print("Tensorflow version " + tf.__version__)
tf.set_random_seed(0)
data_path = "data/"
#data_path = "/home/yue/data/"
img_size = 32
num_channels = 3
img_size_flat = img_size * img_size * num_channels
num_classes = 10
_num_files_train = 5
_images_per_file = 10000
_num_images_train = _num_files_train * _images_per_file
# the path of the saved images
#train_image_store_path="C:/Users/肖岳/Desktop/Final year project/Convolutional Neural Network/train_images/"
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

def _load_data(filename):
    # Load the pickled data-file.
    data = _unpickle(filename)

    # Get the raw images.
    raw_images = data[b'data']

    # Get the class-numbers for each image. Convert to numpy-array.
    labels = data[b'labels']
    images = raw_images
    return images, labels

def load_training_data():
    images=[[0 for i in range(50000)] for j in range(3072)]
    cls=[50000]
    begin = 0
    for i in range(_num_files_train):
        images_batch, cls_batch = _load_data(filename="data_batch_" + str(i + 1))
        num_images = len(images_batch)
        end = begin + num_images
        images[begin:end] = images_batch
        cls[begin:end] = cls_batch
        begin = end
        #onehot_labels=tf.one_hot(indices=tf.cast(cls, tf.int32), depth=10)
    images=np.array(images)# convert list to numpy
    return images, cls

def load_test_data():

    images, cls = _load_data(filename="test_batch")
    #onehot_labels=tf.one_hot(indices=tf.cast(cls, tf.int32), depth=10)
    return images, cls
	

#Xtrain,Ytrain=load_training_data()
#Xtest,Ytest=load_test_data()

def merge_three_channals(te_images,size):
    each_channal=size*size
    te_images=te_images.reshape(each_channal*3)
    img_R = te_images[0:each_channal].reshape((size, size))
    img_G = te_images[each_channal:each_channal*2].reshape((size, size))
    img_B = te_images[each_channal*2:each_channal*3].reshape((size, size))
    te_images = np.dstack((img_R, img_G, img_B))
    return te_images


#save list into txt file
"""file=open('C:/Users/肖岳/Desktop/Final year project/Convolutional Neural Network/train_images/train_labels.txt','w')
file.write(str(Ytrain));
file.close()

file=open('C:/Users/肖岳/Desktop/Final year project/Convolutional Neural Network/test_images/test_labels.txt','w')
file.write(str(Ytest));
file.close()"""
# read data from txt file as list
"""def read_labels
    with open('C:/Users/肖岳/Desktop/Final year project/Convolutional Neural Network/train_images/train_labels.txt', 'r') as f:  
        train_labels=[]
        data = f.readlines() 
        for line in data:  
            odom = line.split(", ")
            train_labels.extend(odom)
            #print (train_labels)
    with open('C:/Users/肖岳/Desktop/Final year project/Convolutional Neural Network/test_images/test_labels.txt', 'r') as f:  
        test_labels=[]
        data = f.readlines()
        for line in data:  
            odom = line.split(", ")
            test_labels.extend(odom)
            #print (train_labels)
    onehot_train_labels=tf.one_hot(indices=tf.cast(train_labels, tf.int32), depth=10)
    onehot_test_labels=tf.one_hot(indices=tf.cast(train_labels, tf.int32), depth=10)
    return onehot_train_labels, onehot_test_labels"""
		
#save picture
"""sess = tf.InteractiveSession()
for i in range(1000):
    te_images= Xtrain[i:i+1].reshape(32,32,3)
    te_images1 = tf.image.resize_images(te_images, [227,227],method=tf.image.ResizeMethod.BILINEAR)
    #te_images2 = tf.image.resize_images(te_images, [227,227],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    #te_images3 = tf.image.resize_images(te_images, [227,227],method=tf.image.ResizeMethod.BICUBIC)
    #te_images4 = tf.image.resize_images(te_images, [227,227],method=tf.image.ResizeMethod.AREA)

    te_images1=(te_images1.eval()).reshape(227,227,3)
    #te_images2=(te_images2.eval()).reshape(227,227,3)
    #te_images3=(te_images3.eval()).reshape(227,227,3)
    #te_images4=(te_images4.eval()).reshape(227,227,3)

    #te_images=merge_three_channals(te_images,32)
    te_images1=merge_three_channals(te_images1,227)
    #te_images2=merge_three_channals(te_images2,227)
    #te_images3=merge_three_channals(te_images3,227)
    #te_images4=merge_three_channals(te_images4,227)

    #im = Image.fromarray(te_images.astype(np.uint8))
    im1 = Image.fromarray(te_images1.astype(np.uint8))
    #im2 = Image.fromarray(te_images2.astype(np.uint8))
    #im3 = Image.fromarray(te_images3.astype(np.uint8))
    #im4 = Image.fromarray(te_images4.astype(np.uint8))

    #im.save("C:/Users/肖岳/Desktop/Final year project/Convolutional Neural Network/train_images/"+str(i+1)+".jpg")
    im1.save("C:/Users/肖岳/Desktop/Final year project/Convolutional Neural Network/train_images/"+str(i+2)+".jpg")
    #im2.save("C:/Users/肖岳/Desktop/Final year project/Convolutional Neural Network/train_images/"+str(i+3)+".jpg")
    #im3.save("C:/Users/肖岳/Desktop/Final year project/Convolutional Neural Network/train_images/"+str(i+4)+".jpg")
    #im4.save("C:/Users/肖岳/Desktop/Final year project/Convolutional Neural Network/train_images/"+str(i+5)+".jpg")



for i in range(500):
    tr_images= Xtest[i].reshape(32,32,3)#reshape
    tr_images = tf.image.resize_images(tr_images, [227,227]) #resize
    tr_images=(tr_images.eval()).reshape(227,227,3) #convert to numpy
    tr_images=merge_three_channals(tr_images,227)  # merge three channels
    im = Image.fromarray(tr_images.astype(np.uint8)) # set data type
    im.save("C:/Users/肖岳/Desktop/Final year project/Convolutional Neural Network/test_images/"+str(i+1)+".jpg")
"""


#read image test
"""
image=Image.open(train_image_store_path+str(2)+".jpg")
r, g, b = image.split()
r_arr = np.array(r).reshape(227*227)
g_arr = np.array(g).reshape(227*227)
b_arr = np.array(b).reshape(227*227)
image_arr = np.concatenate((r_arr, g_arr, b_arr))
print(image_arr)
"""
"""def read_training_images_as_nump(path, begin_from, batch_size, image_size):
    result = np.array([])
    for i in range(batch_size):
        image=Image.open(path+str(begin_from+i)+".jpg")
        img_R, img_G, img_B = image.split()
        r_arr = np.array(img_R).reshape(image_size*image_size)
        g_arr = np.array(img_G).reshape(image_size*image_size)
        b_arr = np.array(img_B).reshape(image_size*image_size)
        image_arr = np.concatenate((r_arr, g_arr, b_arr))
        result = np.concatenate((result, image_arr))
    result = result.reshape((batch_size, image_size*image_size*3))
    return result"""
"""
def next_train_batch(begin, batch_size):
    tr_images=read_training_images_as_nump(train_image_store_path, begin, batch_size, 227)
    tr_images= tr_images.reshape(batch_size,227,227,3)
    tr_labels = train_labels[begin:end]
    return tr_images,tr_labels
"""
