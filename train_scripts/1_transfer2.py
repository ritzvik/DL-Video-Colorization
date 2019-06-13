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
import cv2
K.set_image_data_format('channels_last')

matplotlib.use('Agg')
import matplotlib.pyplot as plt


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


basicmodel = load_model('models/1_basic.h5')



def generator(batchsize, divs, divpart, rseed=69):
    frame_paths = list()
    for framedir in framedirs:
        tmplist = glob.glob(os.path.join(framedir, '*.jpg'))
        frame_paths += [[tmplist[i-2], tmplist[i-1], tmplist[i]] for i in range(2, len(tmplist))]
    ###
    l = len(frame_paths)
    random.seed(rseed)
    random.shuffle(frame_paths)
    frame_paths += frame_paths[-1*int(0.30*l):]
    for i in range(int(0.25*l)):
        frame_paths[-i][0], frame_paths[-i][1] = frame_paths[i][0], frame_paths[i][1]
    ###
    l = len(frame_paths)
    random.seed(rseed)
    random.shuffle(frame_paths)
    frame_paths += frame_paths[-1*int(0.08*l):]
    for i in range(int(0.08*l)):
        frame_paths[-i][0] = '$$'
    ###
    l = len(frame_paths)
    random.seed(rseed)
    random.shuffle(frame_paths)
    ###
    frame_paths = frame_paths[divpart*(l//divs):(divpart+1)*(l//divs)]
    #
    while True:
        for i in range(0, l, batchsize):
            fps = frame_paths[i:i+batchsize]
            inplist, outlist, singlelist = list(), list(), list()
            for f2, f1, f0 in fps:
                try:
                    if f2 != '$$':
                        im2, im1, im0 = cv2.imread(f2), cv2.imread(f1), cv2.imread(f0)
                    else:
                        im2, im1 = np.zeros((x_shape, y_shape, 3), dtype=np.uint8), np.zeros((x_shape, y_shape, 3), dtype=np.uint8)
                        im0 = cv2.imread(f0)
                    im0g = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)
                    inputtensor = np.stack((im2[:,:,0],im2[:,:,1],im2[:,:,2],im1[:,:,0],im1[:,:,1],im1[:,:,2],im0g), axis=-1)
                    inplist.append(inputtensor)
                    singlelist.append(inputtensor[:,:,6:])
                    outlist.append(im0)
                except cv2.error:
                    print('\nCV2 Exception Raised.....ignoring..')
                    print(f0,'\n')
            if len(inplist)==0:
                continue
            inplist, outlist = np.array(inplist)/255.0, np.array(outlist)/255.0
            singlelist = np.array(singlelist)/255.0
            yield {'input_1':inplist,'input_2':singlelist}, outlist



def SubModel_1ch(input_img):
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
    # final output
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same', name = "Ouput_layer")(Upconv3_2)
    #
    submodel = Model(input=input_img, output=[decoded, Upconv1_2, Upconv2_2 ])
    # TRANSFERING THE WEIGHTS from pre-trained model
    if True:
        for replayer, baselayer in zip(submodel.layers[1:], basicmodel.layers[1:]):
            replayer.set_weights(baselayer.get_weights())
            replayer.trainable = False
    ###
    submodel.summary()
    return submodel


def SubModel_7ch(input_img):
    # ENCODER
    Epool1 = MaxPooling2D(pool_size=(2, 2),strides=(2,2),padding='same', name = "block1_pool1")(input_img)
    Econv2_1 = Conv2D(2, (3, 3), activation='relu', padding='same', name = "block2_conv1")(Epool1)
    Econv2_1 = BatchNormalization(name = "Enorm21")(Econv2_1)
    Econv2_2 = Conv2D(2, (5, 5), activation='relu', padding='same', name = "block2_conv2")(Econv2_1)
    Econv2_2 = BatchNormalization(name = "Enorm22")(Econv2_2)
    Epool2= MaxPooling2D(pool_size=(2, 2),strides=(2,2), padding='same', name = "block2_pool1")(Econv2_2)
    Econv3_1 = Conv2D(16, (3, 3), activation='relu', padding='same', name = "block3_conv1")(Epool2)
    Econv3_1 = BatchNormalization(name = "Enorm31")(Econv3_1)
    Econv3_2 = Conv2D(32, (3, 3), activation='relu', padding='same', name = "block3_conv2")(Econv3_1)
    Econv3_2 = BatchNormalization(name = "Enorm32")(Econv3_2)
    Epool3 = MaxPooling2D(pool_size=(2, 2),strides=(2,2), padding='same', name = "block3_pool1")(Econv3_2)
    # NECK
    Nconv1 = Conv2D(64, (3,3),padding = "same", name = "neck1" )(Epool3)
    Nconv1 = BatchNormalization(name = "Nnorm1")(Nconv1)
    Nconv2 = Conv2D(128, (3,3),padding = "same", name = "neck2" )(Nconv1)
    Nconv2 = BatchNormalization(name = "Nnorm2")(Nconv2)
    # DECODER
    up1 = Conv2DTranspose(128,(3,3),strides = (2,2), activation = 'relu', padding = 'same', name = "upsample_1")(Nconv2)
    up1 = BatchNormalization(name = "Dnorm11")(up1)
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
    Upconv3_1 = Conv2D(16, (3, 3), activation='relu', padding='same', name = "Upconv3_1")(up3)
    Upconv3_1 = BatchNormalization(name = "Dnorm32")(Upconv3_1)
    Upconv3_2 = Conv2D(16, (3, 3), activation='relu', padding='same', name = "Upconv3_2")(Upconv3_1)
    Upconv3_2 = BatchNormalization(name = "Dnorm33")(Upconv3_2)
    # final output
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same', name = "Ouput_layer")(Upconv3_2)
    #
    submodel = Model(input=input_img, output=[decoded, Upconv1_2, Upconv2_2 ])
    ###
    submodel.summary()
    return submodel
    


def DualConnect(layers_11, layers_12, layers_21, layers_22, outlayers1, outlayers2):
    concat1 = keras.layers.concatenate([layers_11, layers_12], axis=3, name="dualmerge1")
    conv1 = Conv2D(128, (3,3), activation='relu', padding='same', name='dual_conv1')(concat1)
    conv1 = BatchNormalization()(conv1)

    up2 = Conv2DTranspose(64,(3,3),strides = (2,2), activation = 'relu', padding = 'same', name = "dual_upconv2")(conv1)
    up2 = BatchNormalization()(up2)
    concat2 = keras.layers.concatenate([layers_21, up2, layers_22], axis=3, name="dualmerge2")
    conv2 = Conv2D(64, (3,3), activation='relu', padding='same', name='dual_conv2')(concat2)
    conv2 = BatchNormalization()(conv2)

    up3 = Conv2DTranspose(16,(3,3),strides = (2,2), activation = 'relu', padding = 'same', name = "dual_upconv3")(conv2)
    up3 = BatchNormalization()(up3)
    conv3 = Conv2D(3,(3,3), activation='relu', padding='same', name='dual_conv3')(up3)
    conv3 = BatchNormalization()(conv3)
    concat3 = keras.layers.concatenate([outlayers1, conv3, outlayers2], axis=3, name="dualmerge3")

    conv4 = Conv2D(3, (3,3), activation='sigmoid', padding='same', name='dual_conv4')(concat3)

    connectnet = Model(input = [layers_11, layers_12, layers_21, layers_22, outlayers1, outlayers2],
        output=conv4)
    connectnet.summary()
    return connectnet




############## INITIALIZE ############
x_shape = 360
y_shape = 640
channels = 7
input_img7 = Input(shape = (x_shape, y_shape, channels))
input_img1 = Input(shape = (x_shape, y_shape, 1))


############## GET INTERMEDIATE OUTPUTS FROM SUBMODELS ###############
sub7 = SubModel_7ch(input_img7)
sub1 = SubModel_1ch(input_img1)
out7 = sub7(input_img7)
out1 = sub1(input_img1)


############## CONNECT THE 2 SUB-MODELS ############
outlayers_inp0 = Input(shape = (x_shape, y_shape, 3))
outlayers_inp1 = Input(shape = (x_shape, y_shape, 3))
layers_1_inp0 = Input(shape = (x_shape//2, y_shape//2, 64))
layers_1_inp1 = Input(shape = (x_shape//2, y_shape//2, 64))
layers_2_inp0 = Input(shape = (x_shape//4, y_shape//4, 128))
layers_2_inp1 = Input(shape = (x_shape//4, y_shape//4, 128))
#
DC = DualConnect(layers_2_inp0, layers_2_inp1,
    layers_1_inp0, layers_1_inp1,
    outlayers_inp0, outlayers_inp1)
fout = DC([out7[1], out1[1],
    out7[2], out1[2],
    out7[0], out1[0]])


############# FINAL MODEL #################
fmodel = Model(input = [input_img7, input_img1], output=fout)
fmodel.summary()
fmodel.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])


# SOME PARMETERS
batchsize = 8
big_epochs = 20
epoch_divs = 10

if not os.path.exists('models'):
    os.makedirs('models')
fmodel.save(os.path.join('models','1_transfer2.h5'))


# TRAINING
for epoch in range(big_epochs):
    print('\n\n\n~~~~~~~~MAIN EPOCH #%d  ~~~~~~~\n\n\n'%(epoch))
    for div_number in range(epoch_divs):
        history = fmodel.fit_generator(generator(batchsize, epoch_divs, div_number, epoch+2),
            steps_per_epoch = (no_of_frames//epoch_divs)//batchsize,
            epochs=1,
            verbose=1,
            validation_data=generator(batchsize, epoch_divs, 0, np.random.randint(0,1000)),
            validation_steps=2,
        )
        fmodel.save(os.path.join('models','1_transfer2.h5'))
