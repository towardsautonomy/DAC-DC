from __future__ import print_function
import keras
import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Input, Activation, Concatenate
from keras.layers import Dense, Dropout, Flatten, Reshape
from keras.layers import BatchNormalization
from keras.layers import Conv2D, Convolution2D, MaxPooling2D
from keras.layers import LeakyReLU, ReLU

from keras import backend as K
from keras.models import load_model

from src.configs import *

def DAC_DC(shape=(1242,375,3)):
    inputs = Input(shape)
    
    # Block 1
    layer1 = Convolution2D(filters=16, kernel_size=(1, 1), padding = 'same', kernel_initializer = 'glorot_uniform')(inputs)
    layer1 = Convolution2D(filters=32, kernel_size=(3, 3), padding = 'same', kernel_initializer = 'glorot_uniform')(layer1)
    layer1 = LeakyReLU(alpha=0.1)(layer1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(layer1)

    # Block 2
    layer2 = Convolution2D(filters=32, kernel_size=(3, 3), padding = 'same', kernel_initializer = 'glorot_uniform')(pool1)
    layer2 = BatchNormalization()(layer2)
    layer2 = LeakyReLU(alpha=0.1)(layer2)
    layer2 = Convolution2D(filters=32, kernel_size=(3, 3), padding = 'same', kernel_initializer = 'glorot_uniform')(layer2)
    layer2 = BatchNormalization()(layer2)
    layer2 = LeakyReLU(alpha=0.1)(layer2)
    merge2 = Concatenate(axis = 3)([layer2,pool1])
    pool2 = MaxPooling2D(pool_size=(2, 2))(merge2)

    # Block 3
    layer3 = Convolution2D(filters=64, kernel_size=(3, 3), padding = 'same', kernel_initializer = 'glorot_uniform')(pool2)
    layer3 = BatchNormalization()(layer3)
    layer3 = LeakyReLU(alpha=0.1)(layer3)
    layer3 = Convolution2D(filters=64, kernel_size=(3, 3), padding = 'same', kernel_initializer = 'glorot_uniform')(layer3)
    layer3 = BatchNormalization()(layer3)
    layer3 = LeakyReLU(alpha=0.1)(layer3)
    merge3 = Concatenate(axis = 3)([layer3,pool2])
    pool3 = MaxPooling2D(pool_size=(2, 2))(merge3)

    # Block 4
    layer4 = Convolution2D(filters=128, kernel_size=(3, 3), padding = 'same', kernel_initializer = 'glorot_uniform')(pool3)
    layer4 = BatchNormalization()(layer4)
    layer4 = LeakyReLU(alpha=0.1)(layer4)
    layer4 = Convolution2D(filters=128, kernel_size=(3, 3), padding = 'same', kernel_initializer = 'glorot_uniform')(layer4)
    layer4 = BatchNormalization()(layer4)
    layer4 = LeakyReLU(alpha=0.1)(layer4)
    merge4 = Concatenate(axis = 3)([layer4,pool3])
    pool4 = MaxPooling2D(pool_size=(2, 2))(merge4)

    # Block 5
    layer5 = Convolution2D(filters=256, kernel_size=(3, 3), padding = 'same', kernel_initializer = 'glorot_uniform')(pool4)
    layer5 = BatchNormalization()(layer5)
    layer5 = LeakyReLU(alpha=0.1)(layer5)
    layer5 = Convolution2D(filters=256, kernel_size=(3, 3), padding = 'same', kernel_initializer = 'glorot_uniform')(layer5)
    layer5 = BatchNormalization()(layer5)
    layer5 = LeakyReLU(alpha=0.1)(layer5)
    merge5 = Concatenate(axis = 3)([layer5,pool4])
    pool5 = MaxPooling2D(pool_size=(2, 2))(merge5)

    # Block 6
    layer6 = Convolution2D(filters=512, kernel_size=(3, 3), padding = 'same', kernel_initializer = 'glorot_uniform')(pool5)
    layer6 = BatchNormalization()(layer6)
    layer6 = LeakyReLU(alpha=0.1)(layer6)
    layer6 = Convolution2D(filters=512, kernel_size=(3, 3), padding = 'same', kernel_initializer = 'glorot_uniform')(layer6)
    layer6 = BatchNormalization()(layer6)
    layer6 = LeakyReLU(alpha=0.1)(layer6)
    merge6 = Concatenate(axis = 3)([layer6,pool5])
    pool6 = MaxPooling2D(pool_size=(2, 2))(merge6)

    conv_final = Convolution2D(filters=512, kernel_size=(1, 1), padding = 'same', kernel_initializer = 'glorot_uniform')(pool6)

    # FC Layers
    output_size = n_x_grids*n_y_grids*n_anchors*n_elements_per_grid
    fc1 = Flatten()(conv_final)
    fc2 = Dense(max(output_size, 1024), activation='sigmoid')(fc1)
    out = Dense(output_size, activation='sigmoid')(fc2)

    # model
    model = Model(input = inputs, output = out)
    return model