#!/usr/bin/env python
# coding: utf-8

import keras
from keras.models import *
import numpy as np
import cv2
import glob
import sys
import os


def frameExtractor(path):
    videoObject = cv2.VideoCapture(path)
    success = 1
    while success:
        success, image = videoObject.read()  
        yield image[:,:640,:]/255.0


print("Model loading")
model = load_model(sys.argv[1])
print("Model loaded")
it = frameExtractor('./test_videos/testvid1.mp4')

gt = list()
for i in range(2000):
    im = next(it)
    gt.append(im)
gt = np.array(gt)

preds = np.copy(gt)
abc = preds

for i in range(len(gt)):
    if i%20==0: print(i)
    abc = gt[i]
    gray = cv2.cvtColor((abc*255.0).astype(np.uint8), cv2.COLOR_BGR2GRAY)/255.0
    # inpu = np.stack((preds[i-2][:,:,0],preds[i-2][:,:,1],preds[i-2][:,:,2],preds[i-1][:,:,0],preds[i-1][:,:,1],preds[i-1][:,:,2],gray), axis=-1)
    inpu_1 = gray.reshape((360, 640, 1))
    input_1_list = np.array([inpu_1],dtype=float)
    # input_list = np.array([inpu], dtype=float)
    preds[i] = model.predict(input_1_list)[0]

dirname = sys.argv[1]+'d/'
if not os.path.exists(dirname):
        os.makedirs(dirname)


for i in range(2000):
    temp = np.zeros(shape=(360,640*2,3))
    temp[:,:640,:], temp[:,640:,:] = preds[i], gt[i]
    # temp[:,:,0], temp[:,:,1], temp[:,:,2] = temp[:,:,2], temp[:,:,1], temp[:,:,0]
    cv2.imwrite(dirname+str(i)+'.png',(temp*255).astype(np.uint8))
