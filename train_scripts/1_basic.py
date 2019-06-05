import os
import cv2
import random
import glob
import keras
from keras.layers.core import *
from keras.layers import  Input,Dense,Flatten,Dropout,merge,Reshape,Conv2D,MaxPooling2D,UpSampling2D,Conv2DTranspose,ZeroPadding2D, Add
from keras.layers.normalization import BatchNormalization
from keras.models import Model,Sequential,load_model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adadelta, RMSprop,SGD,Adam
from keras import regularizers
from keras import backend as K
import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt
import cv2
K.set_image_data_format('channels_last')

matplotlib.use('Agg')


print('\n\n\n------------IMPORTS DONE----------\n\n\n')


fileptr = open('train_videos.txt','r')
framedirs = [os.path.join('frames',s) for s in fileptr.read().split('\n')]
fileptr.close()



frame_paths = list()
for framedir in framedirs:
    frame_paths += glob.glob(os.path.join(framedir, '*.jpg'))
no_of_frames = len(frame_paths)
print('Number of frames :',no_of_frames)
del frame_paths



def generator(batchsize, divs, divpart, rseed=69):
    frame_paths = list()
    for framedir in framedirs:
        frame_paths += glob.glob(os.path.join(framedir, '*.jpg'))
    l = len(frame_paths)
    random.seed(rseed)
    random.shuffle(frame_paths)
    frame_paths = frame_paths[divpart*(l//divs):(divpart+1)*(l//divs)]
    l = len(frame_paths)
    while True:
        for i in range(0, l, batchsize):
            fps = frame_paths[i:i+batchsize]
            inplist, outlist = list(), list()
            for fpath in fps:
                try:
                    im = cv2.imread(fpath)
                    gim = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                    outlist.append(im)
                    inplist.append(gim.reshape((gim.shape[0], gim.shape[1], 1)))
                except cv2.error:
                    print('\nCV2 Exception Raised.....ignoring..')
                    print(fpath,'\n')
            if len(inplist)==0:
                continue
            inplist, outlist = np.array(inplist)/255.0, np.array(outlist)/255.0
            yield inplist, outlist


# ------------------------DEFINING THE MODEL--------------------------


x_shape = 360
y_shape = 640
channels = 1

input_img = Input(shape = (x_shape,y_shape,channels), name='3channel_inp')


# ENCODER
Econv1_1 = Conv2D(16, (3, 3), activation='relu', padding='same', name = "block1_conv1")(input_img)
Econv1_1 = BatchNormalization(name = "Enorm11")(Econv1_1)
Econv1_2 = Conv2D(16, (5, 5), activation='relu', padding='same',  name = "block1_conv2")(Econv1_1)
Econv1_2 = BatchNormalization(name = "Enorm12")(Econv1_2)
Epool1 = MaxPooling2D(pool_size=(2, 2),strides=(2,2),padding='same', name = "block1_pool1")(Econv1_2)

Econv2_1 = Conv2D(64, (3, 3), activation='relu', padding='same', name = "block2_conv1")(Epool1)
Econv2_1 = BatchNormalization(name = "Enorm21")(Econv2_1)
Econv2_2 = Conv2D(64, (5, 5), activation='relu', padding='same', name = "block2_conv2")(Econv2_1)
Econv2_2 = BatchNormalization(name = "Enorm22")(Econv2_2)
Epool2= MaxPooling2D(pool_size=(2, 2),strides=(2,2), padding='same', name = "block2_pool1")(Econv2_2)

Econv3_1 = Conv2D(128, (3, 3), activation='relu', padding='same', name = "block3_conv1")(Epool2)
Econv3_1 = BatchNormalization(name = "Enorm31")(Econv3_1)
Econv3_2 = Conv2D(128, (3, 3), activation='relu', padding='same', name = "block3_conv2")(Econv3_1)
Econv3_2 = BatchNormalization(name = "Enorm32")(Econv3_2)
Epool3 = MaxPooling2D(pool_size=(2, 2),strides=(2,2), padding='same', name = "block3_pool1")(Econv3_2)


# NECK
Nconv1 = Conv2D(128, (3,3),padding = "same", name = "neck1" )(Epool3)
Nconv1 = BatchNormalization(name = "Nnorm1")(Nconv1)
Nconv2 = Conv2D(128, (3,3),padding = "same", name = "neck2" )(Nconv1)
Nconv2 = BatchNormalization(name = "Nnorm2")(Nconv2)


# DECODER
up1 = Conv2DTranspose(128,(3,3),strides = (2,2), activation = 'relu', padding = 'same', name = "upsample_1")(Nconv2)
up1 = BatchNormalization(name = "Dnorm11")(up1)
up1 = keras.layers.concatenate([up1, Econv3_2],  axis=3, name = "merge_1")
Upconv1_1 = Conv2D(128, (3, 3), activation='relu', padding='same', name = "Upconv1_1")(up1)
Upconv1_1 = BatchNormalization(name = "Dnorm12")(Upconv1_1)
Upconv1_2 = Conv2D(128, (3, 3), activation='relu', padding='same', name = "Upconv1_2")(Upconv1_1)
Upconv1_2 = BatchNormalization(name = "Dnorm13")(Upconv1_2)

up2 = Conv2DTranspose(64,(3,3),strides = (2,2), activation = 'relu', padding = 'same', name = "upsample_2")(Upconv1_2)
up2 = BatchNormalization(name = "Dnorm21")(up2)
up2 = keras.layers.concatenate([up2, Econv2_2], axis=3, name = "merge_2")
Upconv2_1 = Conv2D(64, (3, 3), activation='relu', padding='same', name = "Upconv2_1")(up2)
Upconv2_1 = BatchNormalization(name = "Dnorm22")(Upconv2_1)
Upconv2_2 = Conv2D(64, (3, 3), activation='relu', padding='same', name = "Upconv2_2")(Upconv2_1)
Upconv2_2 = BatchNormalization(name = "Dnorm23")(Upconv2_2)

up3 = Conv2DTranspose(16,(3,3),strides = (2,2), activation = 'relu', padding = 'same', name = "upsample_3")(Upconv2_2)
up3 = BatchNormalization(name = "Dnorm31")(up3)
up3 = keras.layers.concatenate([up3, Econv1_2], axis=3, name = "merge_3")
Upconv3_1 = Conv2D(16, (3, 3), activation='relu', padding='same', name = "Upconv3_1")(up3)
Upconv3_1 = BatchNormalization(name = "Dnorm32")(Upconv3_1)
Upconv3_2 = Conv2D(16, (3, 3), activation='relu', padding='same', name = "Upconv3_2")(Upconv3_1)
Upconv3_2 = BatchNormalization(name = "Dnorm33")(Upconv3_2)
    
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same', name = "Ouput_layer")(Upconv3_2)


# MODEL
basicmodel = Model(input = input_img, output=decoded)
basicmodel.summary()
basicmodel.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])


# SOME PARMETERS
batchsize = 16
big_epochs = 20
epoch_divs = 10


if not os.path.exists('models'):
    os.makedirs('models')
basicmodel.save(os.path.join('models','1_basic.h5'))


# TRAINING
for epoch in range(big_epochs):
    print('\n\n\n~~~~~~~~MAIN EPOCH #%d  ~~~~~~~\n\n\n'%(epoch))
    for div_number in range(epoch_divs):
        history = basicmodel.fit_generator(generator(batchsize, epoch_divs, div_number, epoch+2),
            steps_per_epoch = (no_of_frames//epoch_divs)//batchsize,
            epochs=1,
            verbose=1,
            validation_data=generator(batchsize, epoch_divs, 0, np.random.randint(0,1000)),
            validation_steps=2,
        )
        # if not os.path.exists('models'):
        #     os.makedirs('models')
        basicmodel.save(os.path.join('models','1_basic.h5'))
