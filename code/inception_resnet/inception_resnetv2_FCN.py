# -*- coding: utf-8 -*-
"""Inception-ResNet V2 model for Keras.
Model naming and structure follows TF-slim implementation (which has some additional
layers and different number of filters from the original arXiv paper):
https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_resnet_v2.py
Pre-trained ImageNet weights are also converted from TF-slim, which can be found in:
https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models
# Reference
- [Inception-v4, Inception-ResNet and the Impact of
   Residual Connections on Learning](https://arxiv.org/abs/1602.07261)
"""

# Origininally created and maintained by F. Chollet as part of Keras application library.
# Modified for use as a fully convolutional network by Jordan Croom
from __future__ import print_function
from __future__ import absolute_import

import warnings

from keras.models import Model
from keras.layers import Activation
from keras.layers import AveragePooling2D
from keras.layers import BatchNormalization
from keras.layers import Concatenate
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import Input
from keras.layers import Lambda
from keras.layers import MaxPooling2D
from keras.utils.data_utils import get_file
from keras.engine.topology import get_source_inputs
from keras import imagenet_utils
from keras.imagenet_utils import _obtain_input_shape
from keras.imagenet_utils import decode_predictions
from keras import backend as K

from keras.applications import inception_resnet_v2


def InceptionResNetV2_FCN(weights = None,
                          input_tensor = None,
                          input_shape = None,
                          classes = 1000):
    """ Initializes the Inception-ResNet v2 architecture based FCN.
    Optionally loads the weights pre-trained on ImageNet for the convolutional layers
    prior to center 1x1 convolutional filter.

    """

    # initialize conv network using built in application library
    InceptionResNetV2_model = InceptionResNetV2(include_top = False,
                                                weights = 'None',
                                                input_tensor = inputs,
                                                input_shape = (256,256,3),
                                                pooling = None,
                                                classes = 3)

    print('Printing Inception Resnet V2 model layers...')
    for layer in InceptionResNetV2_model.layers:
        print(layer.name)
        print('Input shape: {}'.format(layer.input_shape))
        print('Output shape: {}'.format(layer.output_shape))

    """

    # center 1x1 convolutional layer to increase depth
    x = conv2d_batchnorm(InceptionResNetV2_model, 2048, kernel_size = 1, strides = 1)

    ## add in decoder layers with skip connections

    # input size 8x8xN, output 17x17xN
    # upsample center 1x1 conv, add skip connection to output of inception reduction_a ('mixed_6a'), pass through convolution
    x = decoder_block(x, InceptionResNetV2_model.get_layer(name = 'mixed_6a').output, 896, size_mult = (2,2))

    # input size 17x17xN, output 35x35xN
    # upsample result, add skip connection to output of inception A block ('mixed_5b'), pass through convolution
    x = decoder_block(x, InceptionResNetV2_model.get_layer(name = 'mixed_5b').output, 256)

    # input size 35x35xN, output 128x128xN
    # upsample result, pass through convolution
    x = decoder_block(x, None, 128)

    # input size , output 256x256xN
    # upsample result, add skip connection to input
    x = decoder_block(x, inputs, 64)

    # fully connected layer to output image size
    return layers.Conv2D(num_classes, 3, activation='softmax', padding='same')(x)
    """