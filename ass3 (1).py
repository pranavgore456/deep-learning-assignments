# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 07:59:42 2022

@author: prana
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Conv2D,Dropout,Dense, MaxPooling2D
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
input_shape = (28, 28, 1)

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
#print(x_train.dtype)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
#print(x_train.dtype)

x_train = x_train / 255
x_test = x_test / 255
print("shape of training:",x_train.shape)
print("shape of testing:",x_test.shape)


model = Sequential()
model.add(Conv2D(28, kernel_size=(3, 3), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(200, activation = "relu"))
model.add(Dropout(0.3))
model.add(Dense(10, activation = "softmax"))

model.summary()

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,epochs= 2)

test_loss, test_acc = model.evaluate(x_test, y_test)
print("loss=%.3f"%test_loss)
print("Accuracy=%.3f"%test_acc)

image = x_train[0]
plt.imshow(np.squeeze(image), cmap='gray')
plt.show()

image=image.reshape(1, image.shape[0], image.shape[1], image.shape[2])
predict_model = model.predict([image])
print("predicted class: {}".format(np.argmax(predict_model)))
