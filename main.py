import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math
import copy

from keras.models import Sequential, Model
from keras.layers import Input, Dense, Flatten, Dropout, Activation, Lambda, Permute, Reshape
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D
from keras import backend as K
K.set_image_data_format( 'channels_last' ) # WARNING : important for images and tensors dimensions ordering

import cv2
from scipy.io import loadmat
from glob import glob
import os

# Face classifier location
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
        L.append( Convolution2D(cdim, kernel_size=(3, 3), padding='same', activation='relu', name=convname) )
    
    L.append( MaxPooling2D((2, 2), strides=(2, 2)) )
    
    return L

#Creates the neural network model, the overall skeleton. With no weights whatsoever.
def vgg_face_blank():
    
    withDO = True # no effect during evaluation but usefull for fine-tuning
    
    if True:
        mdl = Sequential()
        
        # First layer is a dummy-permutation = Identity to specify input shape
        mdl.add( Permute((1,2,3), input_shape=(224,224,3)) ) # WARNING : 0 is the sample dim

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
        
        mdl.add( Convolution2D(4096, kernel_size=(7, 7), activation='relu', name='fc6') )
        if withDO:
            mdl.add( Dropout(0.5) )
        mdl.add( Convolution2D(4096, kernel_size=(1, 1), activation='relu', name='fc7') )
        if withDO:
            mdl.add( Dropout(0.5) )
        mdl.add( Convolution2D(2622, kernel_size=(1, 1), activation='relu', name='fc8') )
        mdl.add( Flatten() )
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

############################################################# MAIN CODE ##################################################
# Test using CPU
os.environ['CUDA_VISIBLE_DEVICES'] = ''


facemodel = vgg_face_blank()
#facemodel.summary()

#Loads the weights from a .mat file
data = loadmat('vgg-face.mat', matlab_compatible=False, struct_as_record=False)
l = data['layers']
description = data['meta'][0,0].classes[0,0].description

copy_mat_to_keras(facemodel)

# Gets the current path
PATH = os.getcwd()

#Tests all the images in the directory, used to test vggface
for fn in glob(PATH + '/vgg_face_test/*.jpg'):
	image = cv2.imread(fn)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	#detect faces in the image: TER CUIDADO, ESTE ALGORITMO DETECTA TODAS AS CARAS NA IMAGEM - pode ser um problema
	faces = faceCascade.detectMultiScale(gray, 1.3, 5)

	#draw rectangle around the faces:
	for (x,y,w,h) in faces:
	   	cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0),2)

	#Crops the images and resizes it
	crpim = image[y:y+h, x:x+w]
	crpim = cv2.resize(crpim, (224,224))

	pred(facemodel, crpim, transform=False)
	pred(facemodel, crpim, transform=True)