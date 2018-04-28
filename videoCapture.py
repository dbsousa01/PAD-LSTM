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

############################################ Convert a video to sequence of frames #######################################
# Opens a video
PATH = os.getcwd()
VidPath = '/replay/devel/attack/fixed/'

for fn in glob.glob(PATH + VidPath + '*.mov'):
	vidcap = cv2.VideoCapture(fn)
	path = os.path.splitext(os.path.basename(fn))[0]
	frameRate = vidcap.get(5)/3 # Gets the frame rate of the video divided by 5 to obtain 5 frames per second
	#print(frameRate)
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

VidPath = '/replay/devel/real/'

for fn in glob.glob(PATH + VidPath + '*.mov'):
	vidcap = cv2.VideoCapture(fn)
	path = os.path.splitext(os.path.basename(fn))[0]
	frameRate = vidcap.get(5)/3 # Gets the frame rate of the video divided by 5 to obtain 5 frames per second
	#print(frameRate)
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

VidPath = '/replay/devel/attack/hand/'

for fn in glob.glob(PATH + VidPath + '*.mov'):
	vidcap = cv2.VideoCapture(fn)
	path = os.path.splitext(os.path.basename(fn))[0]
	frameRate = vidcap.get(5)/3 # Gets the frame rate of the video divided by 5 to obtain 5 frames per second
	#print(frameRate)
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

VidPath = '/replay/train/attack/hand/'

for fn in glob.glob(PATH + VidPath + '*.mov'):
	vidcap = cv2.VideoCapture(fn)
	path = os.path.splitext(os.path.basename(fn))[0]
	frameRate = vidcap.get(5)/3 # Gets the frame rate of the video divided by 5 to obtain 5 frames per second
	#print(frameRate)
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

VidPath = '/replay/train/attack/fixed/'

for fn in glob.glob(PATH + VidPath + '*.mov'):
	vidcap = cv2.VideoCapture(fn)
	path = os.path.splitext(os.path.basename(fn))[0]
	frameRate = vidcap.get(5)/3 # Gets the frame rate of the video divided by 5 to obtain 5 frames per second
	#print(frameRate)
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

VidPath = '/replay/train/real/'

for fn in glob.glob(PATH + VidPath + '*.mov'):
	vidcap = cv2.VideoCapture(fn)
	path = os.path.splitext(os.path.basename(fn))[0]
	frameRate = vidcap.get(5)/3 # Gets the frame rate of the video divided by 5 to obtain 5 frames per second
	#print(frameRate)
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

VidPath = '/replay/test/attack/hand/'

for fn in glob.glob(PATH + VidPath + '*.mov'):
	vidcap = cv2.VideoCapture(fn)
	path = os.path.splitext(os.path.basename(fn))[0]
	frameRate = vidcap.get(5)/3 # Gets the frame rate of the video divided by 5 to obtain 5 frames per second
	#print(frameRate)
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

VidPath = '/replay/test/attack/fixed/'

for fn in glob.glob(PATH + VidPath + '*.mov'):
	vidcap = cv2.VideoCapture(fn)
	path = os.path.splitext(os.path.basename(fn))[0]
	frameRate = vidcap.get(5)/3 # Gets the frame rate of the video divided by 5 to obtain 5 frames per second
	#print(frameRate)
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

VidPath = '/replay/test/real/'

for fn in glob.glob(PATH + VidPath + '*.mov'):
	vidcap = cv2.VideoCapture(fn)
	path = os.path.splitext(os.path.basename(fn))[0]
	frameRate = vidcap.get(5)/3 # Gets the frame rate of the video divided by 5 to obtain 5 frames per second
	#print(frameRate)
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