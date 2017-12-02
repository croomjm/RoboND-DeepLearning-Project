import os
import glob
import sys
import tensorflow as tf

from scipy import misc
import numpy as np

from tensorflow.contrib.keras.python import keras
from tensorflow.contrib.keras.python.keras import layers, models

from tensorflow import image

from utils import scoring_utils
from utils.separable_conv2d import SeparableConv2DKeras, BilinearUpSampling2D
from utils import data_iterator
from utils import plotting_tools 
from utils import model_tools

def separable_conv2d_batchnorm(input_layer, filters, strides=1):
    output_layer = SeparableConv2DKeras(filters=filters,kernel_size=3, strides=strides,
                             padding='same', activation='relu')(input_layer)
    
    output_layer = layers.BatchNormalization()(output_layer)

    return output_layer

def conv2d_batchnorm(input_layer, filters, kernel_size=3, strides=1):
    output_layer = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, 
                      padding='same', activation='relu')(input_layer)
    
    output_layer = layers.BatchNormalization()(output_layer) 
    return output_layer

def bilinear_upsample(input_layer):
    output_layer = BilinearUpSampling2D((2,2))(input_layer)
    return output_layer

def encoder_block(input_layer, filters, strides):
    # TODO Create a separable convolution layer using the separable_conv2d_batchnorm() function.
    output_layer = separable_conv2d_batchnorm(input_layer, filters, strides)
    
    return output_layer

def decoder_block(small_ip_layer, large_ip_layer, filters):
    
    # TODO Upsample the small input layer using the bilinear_upsample() function.
    upsampled = bilinear_upsample(small_ip_layer)
    
    if large_ip_layer != None:
        # TODO Concatenate the upsampled and large input layers using layers.concatenate
        concatenated = layers.concatenate([upsampled, large_ip_layer])
    
        # TODO Add some number of separable convolution layers
        output_layer = separable_conv2d_batchnorm(concatenated, filters)

    else:
        output_layer = separable_conv2d_batchnorm(upsampled, filters)
    
    return output_layer

def fcn_model(inputs, num_classes):
    # TODO Add Encoder Blocks. 
    # Remember that with each encoder layer, the depth of your model (the number of filters) increases.
    enc1 = encoder_block(inputs, 16, 2)
    enc2 = encoder_block(enc1, 32, 2)
    enc3 = encoder_block(enc2, 64, 2)

    # TODO Add 1x1 Convolution layer using conv2d_batchnorm().
    inner = conv2d_batchnorm(enc3, 128, kernel_size=1, strides=1)
    
    # TODO: Add the same number of Decoder Blocks as the number of Encoder Blocks
    dec1 = decoder_block(inner,enc2, 64)
    dec2 = decoder_block(dec1, enc1, 32)
    dec3 = decoder_block(dec2, inputs, 16)
    
    x=dec3
    
    # The function returns the output layer of your model. "x" is the final layer obtained from the last decoder_block()
    return layers.Conv2D(num_classes, 3, activation='softmax', padding='same')(x)

def train_model(learning_rate, batch_size, num_epochs, steps_per_epoch, validation_steps, workers, optimizer, weight_file_name):
    image_hw = 160
    image_shape = (image_hw, image_hw, 3)
    inputs = layers.Input(image_shape)
    num_classes = 3

    # Call fcn_model()
    output_layer = fcn_model(inputs, num_classes)

    # Define the Keras model and compile it for training
    model = models.Model(inputs=inputs, outputs=output_layer)

    if optimizer == 'Nadam':
        model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss='categorical_crossentropy')
    elif optimizer =='Adam':
        model.compile(optimizer=keras.optimizers.Nadam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004), loss='categorical_crossentropy')
    else:
        raise RuntimeError('Invalid optimizer type "{}"'.format(optimizer))

    # Data iterators for loading the training and validation data
    train_iter = data_iterator.BatchIteratorSimple(batch_size=batch_size,
                                                   data_folder=os.path.join('..', 'data', 'train'),
                                                   image_shape=image_shape,
                                                   shift_aug=True)

    val_iter = data_iterator.BatchIteratorSimple(batch_size=batch_size,
                                                 data_folder=os.path.join('..', 'data', 'validation'),
                                                 image_shape=image_shape)

    logger_cb = plotting_tools.LoggerPlotter(save_graphs = True)
    callbacks = [logger_cb]

    model.fit_generator(train_iter,
                        steps_per_epoch = steps_per_epoch, # the number of batches per epoch,
                        epochs = num_epochs, # the number of epochs to train for,
                        validation_data = val_iter, # validation iterator
                        validation_steps = validation_steps, # the number of batches to validate on
                        callbacks=callbacks,
                        workers = workers)

    # Save your trained model weights
    model_tools.save_network(model, weight_file_name)