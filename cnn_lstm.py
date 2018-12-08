import matplotlib.pyplot as plt
import matplotlib.colors as colors
from PIL import Image
import numpy as np
import math
import copy
import pathlib
import time 
import random
import sys

import tensorflow as tf
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
#from keras.applications.imagenet_utils import preprocess_input
#from imagenet_utils import decode_predictions
from keras import callbacks
from keras import optimizers
from keras.models import Sequential, Model, model_from_json, load_model
from keras.layers import InputLayer, Dense, Flatten, Dropout, Activation, Lambda, Permute, Reshape, TimeDistributed, LSTM
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D, ConvLSTM2D
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras import backend as K
K.set_image_data_format( 'channels_last' ) # WARNING : important for images and tensors dimensions ordering
from keras.regularizers import l2

import cv2
from scipy.io import loadmat
import glob
import os

# Face classifier location - change accordingly
faceCascade = cv2.CascadeClassifier('/home/daniel/opencv-3.3.1/data/haarcascades/haarcascade_frontalface_default.xml')

#################################################### Functions ########################################################
#  VGG-Face follows the structure of VGG16, only the training set is different (and the weights). It has 16 trainable layers,
# 13 convolutional layers which are grouped as 2/3 and then followed by a maxpooling layer. and 3 fully connected layers
# followed by a dropout or flatten function.  

#The model is composed as a sequence of convolutional layers and maxpooling. Function used
def convblock(cdim, nb, bits=3):
    L = []
    
    for k in range(1,bits+1):
        convname = 'conv'+str(nb)+'_'+str(k)
        L.append(TimeDistributed(Convolution2D(cdim,kernel_size=(3,3), padding='same', activation='relu', name=convname) ) )
    
    L.append( TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2)) ) )
    
    return L

#Creates the neural network model, the overall skeleton. With no weights whatsoever.
def vgg_face_blank():
    
    withDO = True # no effect during evaluation but usefull for fine-tuning
    
    if True:
        mdl = Sequential()
        
        # First layer is a dummy-permutation = Identity to specify input shape
        mdl.add(InputLayer(input_shape=(7,224,224,3)) ) # WARNING : 0 is the sample dim

        for l in convblock(64, 1, bits=2):
            mdl.add(l)

        for l in convblock(128, 2, bits=2):
            mdl.add(l)
        
        for l in convblock(256, 3, bits=3):
            mdl.add(l)
            
        for l in convblock(512, 4, bits=3):
            mdl.add(l)
            
        for l in convblock(512, 5, bits=3):
            mdl.add(l)
        
        mdl.add( TimeDistributed(Convolution2D(4096,kernel_size=(7,7), activation='relu', name='fc6') ) )
        if withDO:
            mdl.add( Dropout(0.5) )
        mdl.add( TimeDistributed(Convolution2D(4096,kernel_size=(1,1), activation='relu', name='fc7') ) )
        if withDO:
            mdl.add( Dropout(0.5) )
        mdl.add( TimeDistributed(Convolution2D(2622,kernel_size=(1,1), activation='relu', name='fc8') ) )
        mdl.add( TimeDistributed(Flatten() ) )
        mdl.add( Activation('softmax') )
        
        return mdl
    
    else:
        raise ValueError('not implemented')

# Function that copies the loaded mat file with the weights to the constructed neural network
def copy_mat_to_keras(kmodel):
    kerasnames = [lr.name for lr in kmodel.layers]

    # WARNING : important setting as 2 of the 4 axis have same size dimension
    #prmt = (3,2,0,1) # INFO : for 'th' setting of 'dim_ordering'
    prmt = (0,1,2,3) # INFO : for 'channels_last' setting of 'image_data_format'

    for i in range(l.shape[1]):
        matname = l[0,i][0,0].name[0]
        if matname in kerasnames:
            kindex = kerasnames.index(matname)
            l_weights = l[0,i][0,0].weights[0,0]
            l_bias = l[0,i][0,0].weights[0,1]
            f_l_weights = l_weights.transpose(prmt)
            #f_l_weights = np.flip(f_l_weights, 2) # INFO : for 'th' setting in dim_ordering
            #f_l_weights = np.flip(f_l_weights, 3) # INFO : for 'th' setting in dim_ordering
            assert (f_l_weights.shape == kmodel.layers[kindex].get_weights()[0].shape)
            assert (l_bias.shape[1] == 1)
            assert (l_bias[:,0].shape == kmodel.layers[kindex].get_weights()[1].shape)
            assert (len(kmodel.layers[kindex].get_weights()) == 2)
            kmodel.layers[kindex].set_weights([f_l_weights, l_bias[:,0]])
            
# Function that predicts the input image and outputs the class which has the highest probability
def pred(kmodel, crpimg, transform=False):
    # transform=True seems more robust but I think the RGB channels are not in right order
    imarr = np.array(crpimg).astype(np.float32)

    if transform:
        imarr[:,:,0] -= 129.1863
        imarr[:,:,1] -= 104.7624
        imarr[:,:,2] -= 93.5940
        #
        # WARNING : in this script (https://github.com/rcmalli/keras-vggface) colours are switched
        aux = copy.copy(imarr)
        #imarr[:, :, 0] = aux[:, :, 2]
        #imarr[:, :, 2] = aux[:, :, 0]

        #imarr[:,:,0] -= 129.1863
        #imarr[:,:,1] -= 104.7624
        #imarr[:,:,2] -= 93.5940

    #imarr = imarr.transpose((2,0,1)) # INFO : for 'th' setting of 'dim_ordering'
    imarr = np.expand_dims(imarr, axis=0)

    out = kmodel.predict(imarr)

    best_index = np.argmax(out, axis=1)[0]
    best_name = description[best_index,0]
    print(best_index, best_name[0], out[0,best_index], [np.min(out), np.max(out)])

def replace_intermediate_layer_in_keras(model, layer_id, new_layer):
    from keras.models import Model

    layers = [l for l in model.layers]

    x = layers[0].output
    for i in range(1, len(layers)):
        if i == layer_id:
            x = new_layer(x)
        else:
            x = layers[i](x)

    new_model = Model(input=layers[0].input, output=x)
    return new_model

def insert_intermediate_layer_in_keras(model, layer_id, new_layer):
    from keras.models import Model

    layers = [l for l in model.layers]

    x = layers[0].output
    for i in range(1, len(layers)):
        if i == layer_id:
            x = new_layer(x)
        x = layers[i](x)

    new_model = Model(input=layers[0].input, output=x)
    return new_model

sequence_length = 4

def reset_states(batch, logs):
    if batch % sequence_length == 0:
        model.reset_states()

############################################################# TRAINING DATA ##################################################
#Gets the path for the current working directory
PATH = os.getcwd()
data_real_path = PATH + '/Frames/replay_lstm/train/real/**/*.jpg'
data_attack_path = PATH + '/Frames/replay_lstm/train/attack/**/*.jpg'
val_real_path = PATH + '/Frames/replay_lstm/devel/real/**/*.jpg'
val_attack_path = PATH + '/Frames/replay_lstm/devel/attack/**/*.jpg'
n_videos = 1440
img_data_list = [[] for i in range(n_videos)]
#img_data_list = []
#print(img_data_list)
count = 0
i = 0

for img in sorted(glob.glob(data_real_path, recursive = True)):
	img = image.load_img(os.path.realpath(img), target_size=(224,224))
	x = image.img_to_array(img)
	############################################### TO HSV
	x = x/255 # normalize
	x = colors.rgb_to_hsv(x) # convert
	x[:,:,0] = np.round(x[:,:,0] * 360,0) # Remove the normalization and round the numbers
	x[:,:,1] = np.round(x[:,:,1] * 100,1)
	x[:,:,2] = np.round(x[:,:,2] * 100,1) 
	############################################### TO YCBCR
	#x = cv2.cvtColor(x, cv2.COLOR_RGB2YCrCb) #convert
	##################################################
	x = np.expand_dims(x, axis = 0)
	x = preprocess_input(x)
	#print(x.shape)
	img_data_list[i].append(x)
	count += 1
	if((count % 7) == 0):
		i += 1

print("Nº imagens reais treino: ", count)
count = 0

for img in sorted(glob.glob(data_attack_path, recursive = True)):
	img = image.load_img(os.path.realpath(img), target_size=(224,224))
	x = image.img_to_array(img)
	############################################### TO HSV
	x = x/255 # normalize
	x = colors.rgb_to_hsv(x) # convert
	x[:,:,0] = np.round(x[:,:,0] * 360,0) # Remove the normalization and round the numbers
	x[:,:,1] = np.round(x[:,:,1] * 100,1)
	x[:,:,2] = np.round(x[:,:,2] * 100,1) 
	############################################### TO YCBCR
	#x = cv2.cvtColor(x, cv2.COLOR_RGB2YCrCb) #convert
	##################################################
	x = np.expand_dims(x, axis = 0)
	x = preprocess_input(x) #Try preprocessing.StandardScaler()
	img_data_list[i].append(x)
	count += 1
	if((count % 7) == 0):
		i += 1

print("Nº imagens ataques treino: ",count)


#print(len(img_data_list))
img_data = np.array(img_data_list)
#img_data = img_data.astype('float32')
print (img_data.shape)
img_data=np.rollaxis(img_data,2,0)
print (img_data.shape)
img_data=img_data[0]
print (img_data.shape)

num_classes = 2
num_of_samples = img_data.shape[0]
labels = np.ones((num_of_samples),dtype='int64')

labels[0:240] = 0 #Real
labels[240:1440] = 1 #Spoofing Attack
#print (labels)

names = ['real', 'spoofing attack']
#convert class labels to on-hot encoding
Y = np_utils.to_categorical(labels,num_classes)
#X_train = img_data
#Y_train = Y
X_train,Y_train = shuffle(img_data,Y, random_state=2)

############################################## Validation data ######################################################
n_videos = 1440
img_data_list = [[] for i in range(n_videos)]
count = 0
i = 0

for img in sorted(glob.glob(val_real_path, recursive = True)):
	img = image.load_img(os.path.realpath(img), target_size=(224,224))
	x = image.img_to_array(img)
	############################################### TO HSV
	x = x/255 # normalize
	x = colors.rgb_to_hsv(x) # convert
	x[:,:,0] = np.round(x[:,:,0] * 360,0) # Remove the normalization and round the numbers
	x[:,:,1] = np.round(x[:,:,1] * 100,1)
	x[:,:,2] = np.round(x[:,:,2] * 100,1) 
	############################################### TO YCBCR
	#x = cv2.cvtColor(x, cv2.COLOR_RGB2YCrCb) #convert
	##################################################
	x = np.expand_dims(x, axis = 0)
	x = preprocess_input(x)
	img_data_list[i].append(x)
	count += 1
	if((count % 7) == 0):
		i += 1

print("Nº imagens reais validação: ", count)
count = 0

for img in sorted(glob.glob(val_attack_path, recursive = True)):
	img = image.load_img(os.path.realpath(img), target_size=(224,224))
	x = image.img_to_array(img)
	############################################### TO HSV
	x = x/255 # normalize
	x = colors.rgb_to_hsv(x) # convert
	x[:,:,0] = np.round(x[:,:,0] * 360,0) # Remove the normalization and round the numbers
	x[:,:,1] = np.round(x[:,:,1] * 100,1)
	x[:,:,2] = np.round(x[:,:,2] * 100,1) 
	############################################### TO YCBCR
	#x = cv2.cvtColor(x, cv2.COLOR_RGB2YCrCb) #convert
	##################################################
	x = np.expand_dims(x, axis = 0)
	x = preprocess_input(x) #Try preprocessing.StandardScaler()
	img_data_list[i].append(x)
	count += 1
	if((count % 7) == 0):
		i += 1

print("Nº imagens ataques validação: ",count)

#print(len(img_data_list))
img_data = np.array(img_data_list)
#img_data = img_data.astype('float32')
print (img_data.shape)
img_data=np.rollaxis(img_data,2,0)
print (img_data.shape)
img_data=img_data[0]
print (img_data.shape)

num_classes = 2
num_of_samples = img_data.shape[0]
labels = np.ones((num_of_samples),dtype='int64')

labels[0:240] = 0 #Real
labels[240:1440] = 1 #Spoofing Attack

names = ['real', 'spoofing attack']
#convert class labels to on-hot encoding
Y = np_utils.to_categorical(labels,num_classes)
#X_val = img_data
#Y_val = Y
X_val,Y_val = shuffle(img_data,Y, random_state=2)

del img_data
del labels

#Reshape for the LSTM
#X_train = np.reshape(X_train, (-1,29,224,224,3))
#X_val = np.reshape(X_val, (-1,29,224,224,3))

#Split the dataset into training set and cross-validation set (80-20)
#X_train, X_val, Y_train, Y_val = train_test_split(x,y,test_size=  0.20, random_state = 2, shuffle = True)
print(X_train.shape,Y_train.shape,X_val.shape,Y_val.shape)
#print(Y_train,Y_val)
#print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
############################################################ CREATE AND CHANGE THE NEURAL NETWORK #########################################
# Use this line to run using CPU
#os.environ['CUDA_VISIBLE_DEVICES'] = ''

facemodel = vgg_face_blank()
#facemodel.summary()

#Loads the weights from a .mat file
data = loadmat('/media/daniel/cnn_weights/replay/vgg-face.mat', matlab_compatible=False, struct_as_record=False)
l = data['layers']
description = data['meta'][0,0].classes[0,0].description

copy_mat_to_keras(facemodel)
facemodel.summary()
del data

num_classes = 2
#Changes the fully connected layers
last_layer = facemodel.layers[-4].output
x = ConvLSTM2D(2049, kernel_size=(1,1), activation='relu', name='fc8')(last_layer)
#x = TimeDistributed(Convolution2D(1000,kernel_size=(1,1), activation='relu', name='fc8'))(last_layer)
#x = TimeDistributed(Flatten())(last_layer)
#print(last_layer._keras_shape)
#x = LSTM(30, dropout = 0.2, return_sequences = False)(x)
#print(x._keras_shape)
#x = Dense(num_classes)(x)
x = Dropout(0.2)(x)
x = Flatten()(x)
out = Dense(num_classes, activation = 'softmax')(x) #Try sigmoid if it does not work. It is better for binary classification. it was softmax before
#out = Dense(num_classes, activation = 'linear', W_regularizer=l2(0.01))(x) # For SVM instead of softmax
custom_model = Model(facemodel.input, out)
#custom_model.summary()
del facemodel

#Choose the layers that you want to train
#for layer in custom_model.layers[:-16]:
#	layer.trainable = False

# NOTA: usar StandardScaler para normalizar o sinal de entrada e alterar as funções todas para sigmoid

#Should be low values, lr = 0,0001 decay = 0,0005 e.g
#optimizer = optimizers.SGD(lr = 0.0001, decay =0.000005 , momentum = 0.9,nesterov = True) #lr = 0.01 decay=0.0001: Values from Patch and Depth Base CNN
optimizer = optimizers.Adam(lr = 0.000001) #decay 0.000000005
#optimizer = optimizers.Adam(lr = 0.000001) - best value
custom_model.compile(loss = 'binary_crossentropy', optimizer = optimizer, metrics =['accuracy'])
#custom_model.compile(loss = 'categorical_hinge', optimizer = optimizer, metrics =['accuracy']) # For SVM instead of softmax
custom_model.summary()

earlyStopping = callbacks.EarlyStopping(monitor='val_acc',patience= 10, verbose = 1, mode = 'auto')
modelCheckpoint = callbacks.ModelCheckpoint(filepath = '/media/daniel/cnn_weights/replay/hsv_lstm/my_model.h5', monitor = 'val_acc', verbose = 1, save_best_only = True, mode='auto')
modelCheckpoint2 = callbacks.ModelCheckpoint(filepath = '/media/daniel/cnn_weights/replay/hsv_lstm/my_model_loss.h5', monitor = 'val_loss', verbose = 1, save_best_only = True, mode = 'auto')

t = time.time()
n_epochs = 150
hist = custom_model.fit(X_train,Y_train, batch_size = 1, epochs = n_epochs, callbacks = [earlyStopping,modelCheckpoint,modelCheckpoint2], verbose= 1, validation_data= (X_val, Y_val), shuffle = False)

print('Training time: %s' % (time.time()-t))
(loss, accuracy) = custom_model.evaluate(X_val,Y_val, batch_size=10, verbose=1)

#Save the model and its weights for testing
custom_model.save('/media/daniel/cnn_weights/replay/hsv_lstm/my_model_early.h5')

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