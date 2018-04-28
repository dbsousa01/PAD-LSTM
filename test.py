import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math
import copy
import pathlib
import time 
import random
import sys

from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
#from keras.applications.imagenet_utils import preprocess_input
#from imagenet_utils import decode_predictions
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Flatten, Dropout, Activation, Lambda, Permute, Reshape
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras import backend as K
K.set_image_data_format( 'channels_last' ) # WARNING : important for images and tensors dimensions ordering

import cv2
from scipy.io import loadmat
import glob
import os

faceCascade = cv2.CascadeClassifier('/home/daniel/opencv-3.3.1/data/haarcascades/haarcascade_frontalface_default.xml')


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