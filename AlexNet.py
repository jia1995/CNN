from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
import numpy as np
from keras.utils import np_utils

def AlexNet:
  model = Sequential()# 生成一个model
  model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=X_train.shape[1:]))# C1 卷积层
  model.add(Activation('relu'))# 激活函数：relu, tanh, sigmoid
  model.add(Convolution2D(32, 3, 3))# C2 卷积层
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))# S3 池化
  model.add(Dropout(0.25))# 
  model.add(Convolution2D(64, 3, 3, border_mode='valid')) # C4
  model.add(Activation('relu'))
  model.add(Convolution2D(64, 3, 3)) # C5
  model.add(Activation('relu'))
  model.add(AveragePooling2D(pool_size=(2, 2)))# S6
  model.add(Dropout(0.25))
  model.add(Flatten())# bottleneck 瓶颈
  model.add(Dense(512))# F7 全连接层, 512个神经元
  model.add(Activation('relu'))# 
  model.add(Dropout(0.5))
  model.add(Dense(nb_classes))# label为0~9共10个类别
  model.add(Activation('softmax'))# softmax 分类器
  model.summary() 
  model.compile(loss='cosine', optimizer=Adam(), metrics=['accuracy'])
  return model
