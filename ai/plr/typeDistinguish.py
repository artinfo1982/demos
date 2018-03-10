# coding=utf-8
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPool2D
from keras.optimizers import SGD
from keras import backend as K

K.set_image_dim_ordering('tf')


import cv2
import numpy as np

plateType = [u"蓝牌", u"黄牌", u"新能源车牌", u"白色", u"黑色-港澳"]


def Getmodel_tensorflow(nb_classes):
    img_rows, img_cols = 9, 34
    nb_filters = 32
    nb_pool = 2
    nb_conv = 3

    model = Sequential()
    model.add(Conv2D(16, (5, 5), input_shape=(img_rows, img_cols, 3)))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(nb_pool, nb_pool)))
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


model = Getmodel_tensorflow(5)
model.load_weights("./model/plate_type.h5")
model.save("./model/plate_type.h5")


def SimplePredict(image):
    image = cv2.resize(image, (34, 9))
    image = image.astype(np.float) / 255
    res = np.array(model.predict(np.array([image]))[0])
    return res.argmax()
