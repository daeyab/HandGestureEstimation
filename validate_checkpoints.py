import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from time import sleep

classes = [
    "Thumb Up",
    "Thumb Down",
    "Swiping Up",
    "Swiping Down",
    "Swiping Left",
    "Swiping Right",
    "Stop Sign",
    "No gesture"
    ]
import tensorflow as tf
print(tf.__version__)


# In[3]:


from tensorflow.python.client import device_lib

import os
import math
import pandas as pd
import matplotlib.image as img
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# return gray image
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


hm_frames = 30 # number of frames
# unify number of frames for each training
def get_unify_frames(path):
    offset = 0
    # pick frames
    frames = os.listdir(path)
    frames_count = len(frames)
    # unify number of frames
    if hm_frames > frames_count:
        # duplicate last frame if video is shorter than necessary
        frames += [frames[-1]] * (hm_frames - frames_count)
    elif hm_frames < frames_count:
        # If there are more frames, then sample starting offset
        #diff = (frames_count - hm_frames)
        #offset = diff-1
        frames = frames[0:hm_frames]
    return frames


# In[15]:


# Resize frames
def resize_frame(frame):
    frame = img.imread(frame)
    frame = cv2.resize(frame, (64, 64))
    return frame
    
class Conv3DModel(tf.keras.Model):
  def __init__(self):
    super(Conv3DModel, self).__init__()
    # Convolutions
    self.conv1 = tf.compat.v2.keras.layers.Conv3D(32, (3, 3, 3), activation='relu', name="conv1", data_format='channels_last')
    self.pool1 = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2), data_format='channels_last')
    self.conv2 = tf.compat.v2.keras.layers.Conv3D(64, (3, 3, 3), activation='relu', name="conv1", data_format='channels_last')
    self.pool2 = tf.keras.layers.MaxPool3D(pool_size=(2, 2,2), data_format='channels_last')
    # self.conv3 = tf.compat.v2.keras.layers.Conv3D(128, (3, 3, 3), activation='relu', name="conv3", data_format='channels_last')
    # self.pool3 = tf.keras.layers.MaxPool3D(pool_size=(3, 3, 3), data_format='channels_last')
    # LSTM & Flatten
    self.convLSTM =tf.keras.layers.ConvLSTM2D(40, (3, 3))
    self.flatten =  tf.keras.layers.Flatten(name="flatten")

    # Dense layers
    self.d1 = tf.keras.layers.Dense(128, activation='relu', name="d1")
    self.dropout = tf.keras.layers.Dropout(seed= 10 , rate=  0.5)
    self.out = tf.keras.layers.Dense(8, activation='softmax', name="output")
    

  def call(self, x):
    x = self.conv1(x)
    x = self.pool1(x)
    x = self.conv2(x)
    x = self.pool2(x)
    # x = self.conv3(x)
    # x = self.pool3(x)
    x = self.convLSTM(x)
    x = self.flatten(x)
    x = self.d1(x)
    x = self.dropout(x)
    return self.out(x)

#%%
new_model = Conv3DModel()
#%%
new_model.compile(loss='sparse_categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(),
              metrics = ['accuracy'])
#%%
def normaliz_data(np_data):
    # Normalisation
    scaler = StandardScaler()
    #scaled_images  = normaliz_data2(np_data)
    scaled_images  = np_data.reshape(-1, 30, 64, 64, 1)
    return scaled_images


#%%
new_model.load_weights('1102/1028_cp/cp-0007.ckpt')

classname = "Thumb Down"
#paths ="1000_total_8000_validation/" #검증 데이터 150개 분류
#paths ="test/test01/" #야외 1m 밤
#paths ="test/test02/" #야외 2m 밤
#paths ="test/test03/" #야외 3m 밤
#paths ="test/test04/" #실내 1m 밝음
#paths ="test/test05/" #실내 1m 어두움
#paths ="test/test06/" #실내 3m 밝음
#paths ="test/test07/" #실내 3m 어두움
#paths ="test/test08/" #실내 5m 밝음
#paths ="test/test09/" #실내 5m 어두움
#paths ="test/test10/" #실외 1m 밝음
#paths ="test/test11/" #실외 3m 밝음
paths ="1108_1422/" #실외 5m 밝음
path = paths+classname+"/"

dirs = os.listdir(path)

count = 0
counter_training = 0 # number of training
training_targets = [] # training targets
for directory in dirs:
        new_frame = [] # one training
        # Frames in each folder
        frames = get_unify_frames(path+directory)
        if len(frames) == hm_frames: # just to be sure
            for frame in frames:
                frame = resize_frame(path+directory+'/'+frame)
                new_frame.append(rgb2gray(frame))
                if len(new_frame) == 30: # partition each training on two trainings.
                    # print("=================")
                    # print(directory)
                    counter_training +=1
                    frame_to_predict = np.array(new_frame, dtype=np.float32)
                    frame_to_predict = normaliz_data(frame_to_predict)
                    predict = new_model.predict(frame_to_predict)
                    classe = classes[np.argmax(predict)]
                    #print(classe, 'Precision = ', np.amax(predict)*100,'%')
                    new_frame = []
                    if classe == classname:
                        count +=1
print(classname)
print(count)


classname = "Thumb Up"
path = paths+classname+"/"

dirs = os.listdir(path)

count = 0
counter_training = 0 # number of training
training_targets = [] # training targets
for directory in dirs:
        new_frame = [] # one training
        # Frames in each folder
        frames = get_unify_frames(path+directory)
        if len(frames) == hm_frames: # just to be sure
            for frame in frames:
                frame = resize_frame(path+directory+'/'+frame)
                new_frame.append(rgb2gray(frame))
                if len(new_frame) == 30: # partition each training on two trainings.
                    # print("=================")
                    # print(directory)
                    counter_training +=1
                    frame_to_predict = np.array(new_frame, dtype=np.float32)
                    frame_to_predict = normaliz_data(frame_to_predict)
                    predict = new_model.predict(frame_to_predict)
                    classe = classes[np.argmax(predict)]
                    #print(classe, 'Precision = ', np.amax(predict)*100,'%')
                    new_frame = []
                    if classe == classname:
                        count +=1
print(classname)
print(count)

classname = "Swiping Down"
path = paths+classname+"/"

dirs = os.listdir(path)

count = 0
counter_training = 0 # number of training
training_targets = [] # training targets
for directory in dirs:
        new_frame = [] # one training
        # Frames in each folder
        frames = get_unify_frames(path+directory)
        if len(frames) == hm_frames: # just to be sure
            for frame in frames:
                frame = resize_frame(path+directory+'/'+frame)
                new_frame.append(rgb2gray(frame))
                if len(new_frame) == 30: # partition each training on two trainings.
                    # print("=================")
                    # print(directory)
                    counter_training +=1
                    frame_to_predict = np.array(new_frame, dtype=np.float32)
                    frame_to_predict = normaliz_data(frame_to_predict)
                    predict = new_model.predict(frame_to_predict)
                    classe = classes[np.argmax(predict)]
                    #print(classe, 'Precision = ', np.amax(predict)*100,'%')
                    new_frame = []
                    if classe == classname:
                        count +=1
print(classname)
print(count)

classname = "Swiping Up"
path = paths+classname+"/"

dirs = os.listdir(path)

count = 0
counter_training = 0 # number of training
training_targets = [] # training targets
for directory in dirs:
        new_frame = [] # one training
        # Frames in each folder
        frames = get_unify_frames(path+directory)
        if len(frames) == hm_frames: # just to be sure
            for frame in frames:
                frame = resize_frame(path+directory+'/'+frame)
                new_frame.append(rgb2gray(frame))
                if len(new_frame) == 30: # partition each training on two trainings.
                    # print("=================")
                    # print(directory)
                    counter_training +=1
                    frame_to_predict = np.array(new_frame, dtype=np.float32)
                    frame_to_predict = normaliz_data(frame_to_predict)
                    predict = new_model.predict(frame_to_predict)
                    classe = classes[np.argmax(predict)]
                    #print(classe, 'Precision = ', np.amax(predict)*100,'%')
                    new_frame = []
                    if classe == classname:
                        count +=1
print(classname)
print(count)

classname = "Swiping Left"
path = paths+classname+"/"

dirs = os.listdir(path)

count = 0
counter_training = 0 # number of training
training_targets = [] # training targets
for directory in dirs:
        new_frame = [] # one training
        # Frames in each folder
        frames = get_unify_frames(path+directory)
        if len(frames) == hm_frames: # just to be sure
            for frame in frames:
                frame = resize_frame(path+directory+'/'+frame)
                new_frame.append(rgb2gray(frame))
                if len(new_frame) == 30: # partition each training on two trainings.
                    # print("=================")
                    # print(directory)
                    counter_training +=1
                    frame_to_predict = np.array(new_frame, dtype=np.float32)
                    frame_to_predict = normaliz_data(frame_to_predict)
                    predict = new_model.predict(frame_to_predict)
                    classe = classes[np.argmax(predict)]
                    #print(classe, 'Precision = ', np.amax(predict)*100,'%')
                    new_frame = []
                    if classe == classname:
                        count +=1
print(classname)
print(count)

classname = "Swiping Right"
path = paths+classname+"/"

dirs = os.listdir(path)

count = 0
counter_training = 0 # number of training
training_targets = [] # training targets
for directory in dirs:
        new_frame = [] # one training
        # Frames in each folder
        frames = get_unify_frames(path+directory)
        if len(frames) == hm_frames: # just to be sure
            for frame in frames:
                frame = resize_frame(path+directory+'/'+frame)
                new_frame.append(rgb2gray(frame))
                if len(new_frame) == 30: # partition each training on two trainings.
                    # print("=================")
                    # print(directory)
                    counter_training +=1
                    frame_to_predict = np.array(new_frame, dtype=np.float32)
                    frame_to_predict = normaliz_data(frame_to_predict)
                    predict = new_model.predict(frame_to_predict)
                    classe = classes[np.argmax(predict)]
                    #print(classe, 'Precision = ', np.amax(predict)*100,'%')
                    new_frame = []
                    if classe == classname:
                        count +=1
print(classname)
print(count)

classname = "Stop Sign"
path = paths+classname+"/"

dirs = os.listdir(path)

count = 0
counter_training = 0 # number of training
training_targets = [] # training targets
for directory in dirs:
        new_frame = [] # one training
        # Frames in each folder
        frames = get_unify_frames(path+directory)
        if len(frames) == hm_frames: # just to be sure
            for frame in frames:
                frame = resize_frame(path+directory+'/'+frame)
                new_frame.append(rgb2gray(frame))
                if len(new_frame) == 30: # partition each training on two trainings.
                    # print("=================")
                    # print(directory)
                    counter_training +=1
                    frame_to_predict = np.array(new_frame, dtype=np.float32)
                    frame_to_predict = normaliz_data(frame_to_predict)
                    predict = new_model.predict(frame_to_predict)
                    classe = classes[np.argmax(predict)]
                    #print(classe, 'Precision = ', np.amax(predict)*100,'%')
                    new_frame = []
                    if classe == classname:
                        count +=1
print(classname)
print(count)

classname = "No gesture"
path = paths+classname+"/"

dirs = os.listdir(path)

count = 0
counter_training = 0 # number of training
training_targets = [] # training targets
for directory in dirs:
        new_frame = [] # one training
        # Frames in each folder
        frames = get_unify_frames(path+directory)
        if len(frames) == hm_frames: # just to be sure
            for frame in frames:
                frame = resize_frame(path+directory+'/'+frame)
                new_frame.append(rgb2gray(frame))
                if len(new_frame) == 30: # partition each training on two trainings.
                    # print("=================")
                    # print(directory)
                    counter_training +=1
                    frame_to_predict = np.array(new_frame, dtype=np.float32)
                    frame_to_predict = normaliz_data(frame_to_predict)
                    predict = new_model.predict(frame_to_predict)
                    classe = classes[np.argmax(predict)]
                    #print(classe, 'Precision = ', np.amax(predict)*100,'%')
                    new_frame = []
                    if classe == classname:
                        count +=1
print(classname)
print(count)

new_model.load_weights('training_todaylast/cp-0007.ckpt')

#paths ="1000_total_8000_validation/" #검증 데이터 150개 분류
#paths ="test/test01/" #야외 1m 밤
#paths ="test/test02/" #야외 2m 밤
#paths ="test/test03/" #야외 3m 밤
#paths ="test/test04/" #실내 1m 밝음
#paths ="test/test05/" #실내 1m 어두움
#paths ="test/test06/" #실내 3m 밝음
#paths ="test/test07/" #실내 3m 어두움
#paths ="test/test08/" #실내 5m 밝음
#paths ="test/test09/" #실내 5m 어두움
path ="test/test13/" #실외 1m 밝음
#paths ="test/test11/" #실외 3m 밝음
#paths ="test/test12/" #실외 5m 밝음

dirs = os.listdir(path)

count = 0
counter_training = 0 # number of training
training_targets = [] # training targets
for directory in dirs:
        new_frame = [] # one training
        # Frames in each folder
        frames = get_unify_frames(path+directory)
        if len(frames) == hm_frames: # just to be sure
            for frame in frames:
                frame = resize_frame(path+directory+'/'+frame)
                new_frame.append(rgb2gray(frame))
                if len(new_frame) == 30: # partition each training on two trainings.
                    # print("=================")
                    # print(directory)
                    counter_training +=1
                    frame_to_predict = np.array(new_frame, dtype=np.float32)
                    frame_to_predict = normaliz_data(frame_to_predict)
                    predict = new_model.predict(frame_to_predict)
                    classe = classes[np.argmax(predict)]
                    #print(classe, 'Precision = ', np.amax(predict)*100,'%')
                    new_frame = []
                    if np.amax(predict)*100>=70:
                        count +=1
print("Doing other Things")
print(count)
