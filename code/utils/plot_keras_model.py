import tensorflow as tf
from tensorflow.keras.utils import plot_model
import model_tools
from separable_conv2d import SeparableConv2DKeras, BilinearUpSampling2D
import argparse

parser = argparse.ArgumentParser(description='Input weights file name from which to create a model graph.')
parser.add_argument('weights_file_name',
                    help='Name of the weights file representing the FCN model.')

args = parser.parse_args()

print('Loading model...')
model = model_tools.load_network(args.weights_file_name)
print('Plotting model {}...'.format(args.weights_file_name))
plot_model(model, to_file='../../project_submission/images/graph_{}.png'.format(args.weights_file_name))
print('Done.')