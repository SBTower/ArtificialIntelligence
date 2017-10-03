"""Author: Stuart Tower
"""

import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim


def normalized_columns_initializer(std=1.0):
    """A function to initialize the weights of a network

    :param std: Standard deviation of the initialised weights
    :return: An initializer function that can be used to initialize the weights in the network
    """
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)

    return _initializer


class ConvolutionalNetwork:
    """A basic convolutional neural network architecture
    """
    def __init__(self, scope='global', state_size=None, action_size=None):
        """

        :param scope: The scope (or namespace) to assign the network architecture to
        :param state_size: The size of the state space, used as the size of the input layer to the network
        :param action_size: The size of the action space, used as the size of the output layer of the network
        """
        with tf.variable_scope(scope):
            shape = [None]
            for i in range(len(state_size)):
                shape.append(state_size[i])

            # A placeholder for the input values to the network
            self.inputs = tf.placeholder(shape=shape, dtype=tf.float32)
            # Resize the input image to a square 84x84
            self.imageResize = tf.image.resize_images(self.inputs, [84, 84])
            # Reshape the image to the right format to input to the network
            self.imageIn = tf.reshape(self.imageResize, shape=[-1, 84, 84, 3])
            # Define the layers in the network (here 4 convolutional layers of different sizes)
            self.conv1 = tf.contrib.layers.convolution2d(inputs=self.imageIn, num_outputs=32, kernel_size=[8, 8],
                                                         stride=[4, 4], padding='VALID', biases_initializer=None)
            self.conv2 = tf.contrib.layers.convolution2d(inputs=self.conv1, num_outputs=64, kernel_size=[4, 4],
                                                         stride=[2, 2], padding='VALID', biases_initializer=None)
            self.conv3 = tf.contrib.layers.convolution2d(inputs=self.conv2, num_outputs=64, kernel_size=[3, 3],
                                                         stride=[1, 1], padding='VALID', biases_initializer=None)
            self.conv4 = tf.contrib.layers.convolution2d(inputs=self.conv3, num_outputs=512, kernel_size=[7, 7],
                                                         stride=[1, 1], padding='VALID', biases_initializer=None)
            # Fully connected output layer
            self.outputLayer = tf.contrib.layers.fully_connected(self.conv4, action_size, tf.nn.relu)
            # Get the index of the largest output value
            self.maxOutputNode = tf.argmax(self.outputLayer, 1)


class ConvolutionalDuelingNetwork:
    """A convolutional neural network with the dueling network architecture
    """
    def __init__(self, scope='global', state_size=None, action_size=None):
        with tf.variable_scope(scope):
            shape = [None]
            for i in range(len(state_size)):
                shape.append(state_size[i])

            # A placeholder for the input values ot the network
            self.inputs = tf.placeholder(shape=shape, dtype=tf.float32)
            # Resize the input image to a square 84x84
            self.imageResize = tf.image.resize_images(self.inputs, [84, 84])
            # Reshape the image to the right format to input to the network
            self.imageIn = tf.reshape(self.imageResize, shape=[-1, 84, 84, 3])
            # Define the initial convolutional layers in the network
            self.conv1 = tf.contrib.layers.convolution2d(inputs=self.imageIn, num_outputs=32, kernel_size=[8, 8],
                                                         stride=[4, 4], padding='VALID', biases_initializer=None)
            self.conv2 = tf.contrib.layers.convolution2d(inputs=self.conv1, num_outputs=64, kernel_size=[4, 4],
                                                         stride=[2, 2], padding='VALID', biases_initializer=None)
            self.conv3 = tf.contrib.layers.convolution2d(inputs=self.conv2, num_outputs=64, kernel_size=[3, 3],
                                                         stride=[1, 1], padding='VALID', biases_initializer=None)
            self.conv4 = tf.contrib.layers.convolution2d(inputs=self.conv3, num_outputs=512, kernel_size=[7, 7],
                                                         stride=[1, 1], padding='VALID', biases_initializer=None)

            # Split the output layer into two stream, advantage and value
            self.streamAC, self.streamVC = tf.split(3, 2, self.conv4)
            # Flatten the output into a fully connected layer
            self.streamA = tf.contrib.layers.flatten(self.streamAC)
            self.streamV = tf.contrib.layers.flatten(self.streamVC)
            # Define the variables describing the weights of the fully connected layers for the advantage and value streams
            self.AW = tf.Variable(tf.random_normal([256, action_size]))
            self.VW = tf.Variable(tf.random_normal([256, 1]))
            # Calculate the advantage and value using the weight variables
            self.Advantage = tf.matmul(self.streamA, self.AW)
            self.Value = tf.matmul(self.streamV, self.VW)
            # Combine the advantage and value layers to create a single output layer
            self.outputLayer = self.Value + tf.subtract(self.Advantage,
                                                   tf.reduce_mean(self.Advantage, reduction_indices=1, keep_dims=True))
            # Get the index of the largest output value
            self.maxOutputNode = tf.argmax(self.outputLayer, 1)


class ConvolutionalActorCriticNetwork:
    """A convolutional actor-critic network
    """
    def __init__(self, scope='global', state_size=None, action_size=None):
        with tf.variable_scope(scope):
            # A placeholder for the input values to the network
            self.inputs = tf.placeholder(shape=[None, state_size], dtype=tf.float32)
            # Resize the input to an 84x84 image
            self.imageResize = tf.image.resize_images(self.inputs, [84, 84])
            # Convert the image to a format to be input to the network
            self.imageIn = tf.reshape(self.inputs, shape=[-1, 84, 84, 1])
            # Define the initial convolutional layers
            self.conv1 = slim.conv2d(activation_fn=tf.nn.elu, inputs=self.imageIn, num_outputs=16, kernel_size=[8, 8],
                                     stride=[4, 4], padding='VALID')
            self.conv2 = slim.conv2d(activation_fn=tf.nn.elu, inputs=self.conv1, num_outputs=32, kernel_size=[4, 4],
                                     stride=[2, 2], padding='VALID')
            # Convert to a fully_connected layer
            hidden = slim.fully_connected(slim.flatten(self.conv2), 256, activation_fn=tf.nn.elu)

            # The policy layer acts as the 'actor', where the output is the action values
            self.policyLayer = slim.fully_connected(hidden, action_size, activation_fn=tf.nn.softmax,
                                                    weights_initializer=normalized_columns_initializer(0.01),
                                                    biases_initializer=None)
            # The value layer acts as the 'critic', where the output is the value of the actions
            self.valueLayer = slim.fully_connected(hidden, 1, activation_fn=None,
                                                   weights_initializer=normalized_columns_initializer(1.0),
                                                   biases_initializer=None)
