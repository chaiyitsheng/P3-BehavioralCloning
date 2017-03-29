import os
import csv
import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Flatten,Dense,Lambda,Convolution2D,MaxPooling2D, Cropping2D

print("Running...")    
lines = []
filepath=".././testLap1/"

with open('.././testLap1/driving_log.csv') as cfile:
          reader = csv.reader(cfile)
          for line in reader:
              lines.append(line)

images=[]
measurements=[]


'''
Reminder!

mpimg.imread reads in images in RGB format
cv2.imread reads in images in BGR format
Please note the difference when using the drive.py file 
'''

# in the loops below:
#x[0],x[1],x[2],x[3] corresponds to centre img, right img, left img and steering angle respectively
# (x[3]!="0") helps ignore datapoints with zero sterring angles

for x in lines:
    if (x[3]!="0"):
        image_path=x[0]
        image=cv2.imread(image_path,cv2.IMREAD_COLOR)
        images.append(image)
        measurements.append(float(x[3]))
    else:
        pass

for x in lines:
    if (x[3]!="0"):
        image_path=x[1]
        image=cv2.imread(image_path,cv2.IMREAD_COLOR)
        images.append(image)
        measurements.append(float(x[3]))
    else:
        pass


for x in lines:
    if (x[3]!="0"):
        image_path=x[2]
        image=cv2.imread(image_path,cv2.IMREAD_COLOR)
        images.append(image)
        measurements.append(float(x[3]))
    else:
        pass

print("Images Completed...")
print("Images:")
print(len(images))
print("Measurements:")
print(len(measurements))
   
x_train = np.array(images)
y_train = np.array(measurements)


# Model Starts here
# This is based on the Nvidia Model but the input image is cropped smaller (66x160x3) that the original research paper(66x200x3)

model=Sequential()

#Preprocessing 1: Lambda Layer to normalize(/255) and mean-center(-0.5) the data
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))

#Preprocessing2 :Cropping the image to reduce size and irrelevant features
model.add(Cropping2D(cropping=((74,20),(80,80))))

#NVidia Model
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,1,8,activation="relu"))
model.add(Flatten())
model.add(Dense(1164))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train,validation_split=0.01,shuffle=True,nb_epoch=5)
model.summary()
model.save('model.h5')