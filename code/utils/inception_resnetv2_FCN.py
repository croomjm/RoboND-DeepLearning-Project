from .inception_resnet_v2 import InceptionResNetV2


def InceptionResNetV2_FCN(shape,
                          num_classes,
                          model_weights = 'None',
                          inputs = None,
                          ):
    """ Initializes the Inception-ResNet v2 architecture based FCN.
    Optionally loads the weights pre-trained on ImageNet for the convolutional layers
    prior to center 1x1 convolutional filter.

    """

    # initialize conv network using built in application library
    InceptionResNetV2_model = InceptionResNetV2(include_top = False,
                                                weights = model_weights,
                                                input_tensor = inputs,
                                                input_shape = shape,
                                                pooling = None,
                                                classes = num_classes)

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

if __name__ == '__main__':
    InceptionResNetV2_FCN((256,256,3), 3)