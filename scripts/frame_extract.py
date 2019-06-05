# put all the training videos of 640X360 in root folder with name "train_videos"

import cv2
import sys
import os
import glob
from os import listdir

vidpath = 'train_videos/'
framepath = 'frames/'
vidfiles = [f for f in listdir(vidpath) if os.path.isfile(os.path.join(vidpath, f))]
print(vidfiles)

def frameExtractor(filename):
    print('\n\n',filename,'\n\n')
    ##
    videoObject = cv2.VideoCapture(os.path.join(vidpath, filename))
    framedir = os.path.join(framepath, filename)
    if not os.path.exists(framedir):
        os.makedirs(framedir)
    
    count = 1
    success = 1
    while success:
        success, image = videoObject.read()
        cv2.imwrite(os.path.join(framedir, "frame%d.jpg"%(count)), image) 
        count += 1
        if count%1000==0:
            print(count)

    fileptr = open('train_videos.txt','a')
    fileptr.write(filename+'\n')
    fileptr.close()

    return count



for vidfile in vidfiles:
    try:
        frameExtractor(vidfile)
    except:
        pass
