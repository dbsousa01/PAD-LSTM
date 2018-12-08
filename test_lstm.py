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

from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
#from keras.applications.imagenet_utils import preprocess_input
#from imagenet_utils import decode_predictions
from keras import optimizers
from keras.models import Sequential, Model, load_model
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
import bob.measure

faceCascade = cv2.CascadeClassifier('/home/daniel/opencv-3.3.1/data/haarcascades/haarcascade_frontalface_default.xml')

# Use this line to run using CPU
#os.environ['CUDA_VISIBLE_DEVICES'] = ''

PATH = os.getcwd()
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

####################################################### Test Replay-Attack DB #########################################

test_real_path = PATH + '/Frames/casia-fasd/train/real/**/*.jpg'
test_attack_path = PATH + '/Frames/casia-fasd/train/ataque/**/*.jpg'

model = load_model('/media/daniel/cnn_weights/replay/hsv_lstm/my_model_all_time7.h5')
model.summary()
n_videos = 360 + 1079
img_test_list = [[] for i in range(n_videos)]
TN = 0
TP = 0
count = 0
i = 0

#out2 = np.load('out_hsv.npy')
for img in sorted(glob.glob(test_real_path, recursive = True)):
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
    img_test_list[i].append(x)
    count += 1
    if((count % 7) == 0):
        i += 1
    TN += 1

print("Nº Reais =",TN)
count = 0

for img in sorted(glob.glob(test_attack_path, recursive = True)):
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
    img_test_list[i].append(x)
    count += 1
    if((count % 7) == 0):
        i += 1
    TP += 1

print("Nº Ataques = ",TP)

img_test = np.array(img_test_list)
#img_data = img_data.astype('float32')
print (img_test.shape)
img_test=np.rollaxis(img_test,2,0)
print (img_test.shape)
img_test=img_test[0]
print (img_test.shape)

num_of_samples = img_test.shape[0]
labels = np.ones((num_of_samples),dtype='int64')
np.set_printoptions(threshold= np.nan)

labels[0:360] = 0 #Real
labels[360:1439] = 1 #Spoofing Attack

out = model.predict(img_test, verbose = 1, batch_size = 4)
#out = np.maximum(out,out2)
best_index = np.argmax(out, axis=1)

FP = 0
TN = 0
FN = 0
TP = 0
acertou = 0
for idx, val in enumerate(out):
    print(str(idx),out[idx])
    best_index = np.argmax(val)
    if labels[idx] == 0 and  best_index == 1: #Caso de ser falso positivo
        FP += 1
        print(idx)
        print(val)
    elif labels[idx] == 0 and best_index == 0: #Caso de ser true negative
        TN +=1
        acertou += 1
    elif labels[idx] == 1 and best_index == 1: #Caso de ser true positive
        TP +=1
        acertou +=1
    elif labels[idx] == 1 and best_index == 0: #Caso de ser falso negativo - mais grave 
        FN += 1
        print(idx)
        print(val)

acc = acertou/num_of_samples
print("Accuracy: ",acc)
FRR = FP / (FP+TN)
print("FRR: ", FRR)
FAR = FN / (FN + TP)
print("FAR: ", FAR)
HTER = (FAR + FRR) / 2
print("HTER: ", HTER)
#np.save('out_hsv',out)

"""
bob.measure.plot.det(FRR, FAR, 100, color=(0,0,0), linestyle='-', label='test') 
bob.measure.plot.det_axis([0.01, 40, 0.01, 40]) 
plt.xlabel('FAR (%)') 
plt.ylabel('FRR (%)') 
plt.grid(True)
plt.save('plotDET.pdf', bbox_inches='tight') 

eer1 = bob.measure.eer_rocch(FRR, FAR)
print("EER = ",eer1)
"""

K.clear_session()