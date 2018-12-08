import matplotlib.pyplot as plt
import matplotlib.colors as colors
from PIL import Image
import numpy as np
import copy
import time 
import h5py
import sys

import tensorflow as tf
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
#from keras.applications.imagenet_utils import preprocess_input
#from imagenet_utils import decode_predictions
from keras import callbacks
from keras import optimizers
from keras.models import Sequential, Model, load_model
from keras.layers import InputLayer, Dense, Flatten, Dropout, Activation, Lambda, Permute, TimeDistributed
from keras.layers import Convolution2D, MaxPooling2D, ConvLSTM2D, concatenate, Input
from keras.utils import np_utils
from sklearn.utils import shuffle
from keras import backend as K
K.set_image_data_format( 'channels_last' ) # WARNING : important for images and tensors dimensions ordering

import cv2
from scipy.io import loadmat
import glob
import os

# Face classifier location
faceCascade = cv2.CascadeClassifier('/home/daniel/opencv-3.3.1/data/haarcascades/haarcascade_frontalface_default.xml')

def generator(file_hsv, file_ycbcr, batch_size):

    train_vector1 = np.zeros((batch_size, 7, 224, 224 ,3 ))
    train_vector2 = np.zeros((batch_size, 7, 224, 224 ,3 ))
    label = np.zeros((batch_size, 2))

    indexes = np.random.choice(len(h5f_hsv['train_hsv']),batch_size)
    indexes.sort()
    train_vector1 = h5f_hsv['train_hsv'][np.array(indexes).tolist()]
    train_vector2 = h5f_ycbcr['train_ycbcr'][np.array(indexes).tolist()]
    label = h5f_hsv['train_label'][np.array(indexes).tolist()]

    yield ([train_vector1, train_vector2] , label)

# TensorFlow wizardry
config = tf.ConfigProto()
 
# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True
K.tensorflow_backend.set_session(tf.Session(config=config))

Input1 = Input(shape=(2049,))
Input2 = Input(shape=(2049,))

merged = concatenate([Input1, Input2])
out = Dense(2, activation = 'softmax')(merged)

fusion_model = Model([Input1, Input2], out)
#Should be low values, lr = 0,0001 decay = 0,0005 e.g
#optimizer = optimizers.SGD(lr = 0.0001, decay =0.000005 , momentum = 0.9,nesterov = True) #lr = 0.01 decay=0.0001: Values from Patch and Depth Base CNN
optimizer = optimizers.Adam(lr = 0.000001) #decay 0.000000005
#optimizer = optimizers.Adam(lr = 0.000001) - best value
fusion_model.compile(loss = 'binary_crossentropy', optimizer = optimizer, metrics =['accuracy'])
#custom_model.compile(loss = 'categorical_hinge', optimizer = optimizer, metrics =['accuracy']) # For SVM instead of softmax
fusion_model.summary()

earlyStopping = callbacks.EarlyStopping(monitor='val_loss',patience= 10, verbose = 1, mode = 'auto')
modelCheckpoint = callbacks.ModelCheckpoint(filepath = '/media/daniel/cnn_weights/replay/hsv_lstm/my_model.h5', monitor = 'val_acc', verbose = 1, save_best_only = True, mode='auto')
modelCheckpoint2 = callbacks.ModelCheckpoint(filepath = '/media/daniel/cnn_weights/replay/hsv_lstm/my_model_loss.h5', monitor = 'val_loss', verbose = 1, save_best_only = True, mode = 'auto')

t = time.time()
n_epochs = 200
#hist = fusion_model.fit([X_train1, X_train2],Y_train1, batch_size = 1, epochs = n_epochs, callbacks = [earlyStopping,modelCheckpoint,modelCheckpoint2], verbose= 1, validation_data= ([X_val1, X_val2],Y_val1), shuffle = False)
h5f_hsv = h5py.File('data_fusion_hsv.h5','r') #Opens the data file
h5f_ycbcr = h5py.File('data_fusion_ycbcr.h5', 'r')

#hist = fusion_model.fit_generator(generator(h5f_hsv, h5f_ycbcr,10), steps_per_epoch = 144, epochs=n_epochs, verbose = 1, callbacks = [earlyStopping,modelCheckpoint,modelCheckpoint2],validation_data= ([h5f_hsv['val_hsv'], h5f_ycbcr['val_ycbcr']],h5f_hsv['val_label']), shuffle = False, initial_epoch = 0)
hist = fusion_model.fit([h5f_hsv['train_hsv'][:], h5f_ycbcr['train_ycbcr'][:]],h5f_hsv['train_label'][:], batch_size = 10, epochs = n_epochs, callbacks = [earlyStopping,modelCheckpoint,modelCheckpoint2], verbose= 1, validation_data= ([h5f_hsv['val_hsv'][:], h5f_ycbcr['val_ycbcr'][:]],h5f_hsv['val_label'][:]), shuffle = False)
print('Training time: %s' % (time.time()-t))
(loss, accuracy) = fusion_model.evaluate([h5f_hsv['val_hsv'][:], h5f_ycbcr['val_ycbcr'][:]],h5f_hsv['val_label'][:], batch_size=10, verbose=1)

h5f_hsv.close() # Closes the data file
h5f_ycbcr.close()
#Save the model and its weights for testing
fusion_model.save('/media/daniel/cnn_weights/replay/hsv_lstm/my_model_early.h5')

print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))

######################################################################## PLOTS #########################################################
# 
#
#VER ISTO PARA JUSTIFICAR A FALTA DE OVERFITTING: https://stackoverflow.com/questions/45135551/validation-accuracy-is-always-greater-than-training-accuracy-in-keras
#
# visualizing losses and accuracy
train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['acc']
val_acc=hist.history['val_acc']
xc=len(train_acc)

plt.figure(1,figsize=(7,5))
plt.plot(train_acc)
plt.plot(val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('Model Accuracy')
plt.grid(color = '#7f7f7f', linestyle = 'dotted')
plt.legend(['train','val'], loc= 0)
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])
plt.savefig('plot1.pdf', bbox_inches='tight')
plt.close()

plt.figure(1,figsize=(7,5))
plt.plot(train_loss)
plt.plot(val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('Model Loss')
plt.grid(color = '#7f7f7f',linestyle = 'dotted')
plt.legend(['train','val'] , loc = 0)
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])
plt.savefig('plot2.pdf', bbox_inches='tight')
plt.close()

"""
# list all data in history
print(hist.history.keys())
# summarize history for accuracy
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('plot1.pdf', bbox_inches='tight')
# summarize history for loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('plot2.pdf', bbox_inches='tight')
"""
K.clear_session()