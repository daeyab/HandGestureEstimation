import os
import tensorflow as tf
import numpy as np
import cv2
from sklearn.preprocessing import StandardScaler
from time import sleep

#8 gesture classes to be classified
gestures = [
    "Thumb Up",
    "Thumb Down",
    "Swiping Up",
    "Swiping Down",
    "Swiping Left",
    "Swiping Right",
    "Stop Sign",
    "No gesture"
    ]
#convert RGB to gray scale
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

#normailize numpy data
def normaliz_data(np_data):
    scaler = StandardScaler()
    scaled_images  = np_data.reshape(-1, 30, 64, 64, 1)
    return scaled_images

#Convolution 3D Model
class Conv3DModel(tf.keras.Model):
  def __init__(self):
    super(Conv3DModel, self).__init__()
    # Convolutions
    conv1_filter = 32
    conv2_filter = 64
    self.conv1 = tf.compat.v2.keras.layers.Conv3D(conv1_filter, (3, 3, 3), activation='relu', name="conv1", data_format='channels_last')
    self.pool1 = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2), data_format='channels_last')
    self.conv2 = tf.compat.v2.keras.layers.Conv3D(conv2_filter, (3, 3, 3), activation='relu', name="conv2", data_format='channels_last')
    self.pool2 = tf.keras.layers.MaxPool3D(pool_size=(2, 2,2), data_format='channels_last')
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
new_model.load_weights('1108_cp_cp/1108_weights')
#new_model.load_weights('final/1024_1742')ㄴ
#%%

to_predict = []
num_frames = 0
cap = cv2.VideoCapture(0)
classe =''


predict =0.0
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    to_predict.append(cv2.resize(gray, (64, 64)))
    
    if len(to_predict) == 30:
        frame_to_predict = np.array(to_predict, dtype=np.float32)
        frame_to_predict = normaliz_data(frame_to_predict)
        #print(frame_to_predict)
        predict = new_model.predict(frame_to_predict)
        classe = gestures[np.argmax(predict)]
        print
    
        print('Classe = ',classe, 'Precision = ', np.amax(predict)*100,'%')


        #print(frame_to_predict)
        to_predict = []
        #sleep(0.1) # Time in seconds
        #font = cv2.FONT_HERSHEY_SIMPLEX
        

#    if np.amax(predict)*100>=90:
    cv2.putText(frame, classe, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0),1,cv2.LINE_AA)


    # Display the resulting frame
    cv2.imshow('Hand Gesture Recognition',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
