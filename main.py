import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math
import copy
import pathlib
import time 

from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
#from keras.applications.imagenet_utils import preprocess_input
#from imagenet_utils import decode_predictions
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Flatten, Dropout, Activation, Lambda, Permute, Reshape
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from keras import backend as K
K.set_image_data_format( 'channels_last' ) # WARNING : important for images and tensors dimensions ordering

import cv2
from scipy.io import loadmat
import glob
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

############################################################ CREATE AND CHANGE THE NEURAL NETWORK #########################################
# Use this line to run using CPU
#os.environ['CUDA_VISIBLE_DEVICES'] = ''

facemodel = vgg_face_blank()
#facemodel.summary()

#Loads the weights from a .mat file
data = loadmat('vgg-face.mat', matlab_compatible=False, struct_as_record=False)
l = data['layers']
description = data['meta'][0,0].classes[0,0].description

copy_mat_to_keras(facemodel)
facemodel.summary()

facemodel = replace_intermediate_layer_in_keras(facemodel, 23,Convolution2D(2, kernel_size=(1, 1), activation='relu', name='fc8') )
facemodel.add( Activation('softmax') )
facemodel.summary()
############################################################# TRAINING DATA ##################################################

# Gets the path for the current working directory
PATH = os.getcwd()
data_real_path = PATH + '/Frames/replay/train/real/**/*.jpg'
data_attack_path = PATH + '/Frames/replay/train/attack/**/*.jpg'
img_data_list = []

"""
for img in glob.glob(data_real_path, recursive = True):
	img = image.load_img(os.path.realpath(img), target_size=(224,224))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis = 0)
	x = preprocess_input(x)
	img_data_list.append(x)

#print(len(img_data_list))

for img in glob.glob(data_attack_path, recursive = True):
	img = image.load_img(os.path.realpath(img), target_size=(224,224))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis = 0)
	x = preprocess_input(x)
	img_data_list.append(x)

#print(len(img_data_list))
img_data = np.array(img_data_list)
#img_data = img_data.astype('float32')
print (img_data.shape)
img_data=np.rollaxis(img_data,1,0)
print (img_data.shape)
img_data=img_data[0]
print (img_data.shape)

num_classes = 2
num_of_samples = img_data.shape[0]
labels = np.ones((num_of_samples),dtype='int64')

labels[0:8999] = 0 #Real
labels[9000:37199] = 1 #Spoofing Attack

names = ['Real', 'Spoofing Attack']
#convert class labels to on-hot encoding
Y = np_utils.to_categorical(labels,num_classes)

#Shuffle the dataset
x,y = shuffle(img_data,Y,random_state=2)
#Split the dataset into training set and cross-validation set (80-20)
X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size=  0.2, random_state = 2) 

"""
"""
############################################ Convert a video to sequence of frames #######################################
# Opens a video
VidPath = '/replay/devel/attack/hand/'

for fn in glob(PATH + VidPath + '*.mov'):
	vidcap = cv2.VideoCapture(fn)
	path = os.path.splitext(os.path.basename(fn))[0]
	frameRate = vidcap.get(5)/5 # Gets the frame rate of the video divided by 5 to obtain 5 frames per second
	#print("Frame Rate:%d , %d" % (frameRate, math.floor(frameRate)))
	count = 1

	#Creates a directory to save image frames
	pathlib.Path(PATH + '/Frames' + VidPath + path).mkdir(parents=True, exist_ok=True) #Creates a directory to save the frames
		
	while(vidcap.isOpened()):
		frameID = vidcap.get(1) # Gets the current frame number
		success, image = vidcap.read()
		if( success != True):
			break
		if((frameID % math.floor(frameRate)) == 0):
			gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			#detect faces in the image: TER CUIDADO, ESTE ALGORITMO DETECTA TODAS AS CARAS NA IMAGEM - pode ser um problema
			faces = faceCascade.detectMultiScale(image, 1.3, 5)

			#draw rectangle around the faces:
			for (x,y,w,h) in faces:
			   	cv2.rectangle(image, (x,y), (x+w,y+h), (0,0,0),0)


			try:
				(x,y,w,h)
			except NameError:
				print('Oops some variable was not defined, face not detected')
			else:
				#Crops the images and resizes it
				crpim = image[y:y+h, x:x+w]
				crpim = cv2.resize(crpim, (224,224))

				cv2.imwrite(PATH + '/Frames' + VidPath + path + '/frame_%d.jpg' % count, crpim) # Save frame as a jpeg file
				success,image = vidcap.read()
				#print('Read a new frame: ', success, frameID)
				count += 1
"""
########################################################## Test VGG-Face code ##########################################
"""
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
	#pred(facemodel, crpim, transform=True) Better for low res images (?)
"""
######################################################################## PLOTS #########################################################

# visualizing losses and accuracy
"""
train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['acc']
val_acc=hist.history['val_acc']
xc=range(12)

plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','val'])
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])
savefig('plot1.pdf', bbox_inches='tight')

plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train','val'],loc=4)
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])
savefig('plot2.pdf', bbox_inches='tight')

"""

K.clear_session()