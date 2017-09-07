import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim


def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)

    return _initializer


class ConvolutionalNetwork:
    def __init__(self, scope='global', state_size=None, action_size=None):
        with tf.variable_scope(scope):
            shape = [None]
            for i in range(len(state_size)):
                shape.append(state_size[i])

            self.inputs = tf.placeholder(shape=shape, dtype=tf.float32)
            self.imageResize = tf.image.resize_images(self.inputs, [84, 84])
            self.imageIn = tf.reshape(self.imageResize, shape=[-1, 84, 84, 3])

            self.conv1 = tf.contrib.layers.convolution2d(inputs=self.imageIn, num_outputs=32, kernel_size=[8, 8],
                                                         stride=[4, 4], padding='VALID', biases_initializer=None)
            self.conv2 = tf.contrib.layers.convolution2d(inputs=self.conv1, num_outputs=64, kernel_size=[4, 4],
                                                         stride=[2, 2], padding='VALID', biases_initializer=None)
            self.conv3 = tf.contrib.layers.convolution2d(inputs=self.conv2, num_outputs=64, kernel_size=[3, 3],
                                                         stride=[1, 1], padding='VALID', biases_initializer=None)
            self.conv4 = tf.contrib.layers.convolution2d(inputs=self.conv3, num_outputs=512, kernel_size=[7, 7],
                                                         stride=[1, 1], padding='VALID', biases_initializer=None)

            self.outputLayer = tf.contrib.layers.fully_connected(self.conv4, action_size, tf.nn.relu)
            self.maxOutputNode = tf.argmax(self.outputLayer, 1)


class ConvolutionalDuelingNetwork:
    def __init__(self, scope='global', state_size=None, action_size=None):
        with tf.variable_scope(scope):
            shape = [None]
            for i in range(len(state_size)):
                shape.append(state_size[i])

            self.inputs = tf.placeholder(shape=shape, dtype=tf.float32)
            self.imageResize = tf.image.resize_images(self.inputs, [84, 84])
            self.imageIn = tf.reshape(self.imageResize, shape=[-1, 84, 84, 3])

            self.conv1 = tf.contrib.layers.convolution2d(inputs=self.imageIn, num_outputs=32, kernel_size=[8, 8],
                                                         stride=[4, 4], padding='VALID', biases_initializer=None)
            self.conv2 = tf.contrib.layers.convolution2d(inputs=self.conv1, num_outputs=64, kernel_size=[4, 4],
                                                         stride=[2, 2], padding='VALID', biases_initializer=None)
            self.conv3 = tf.contrib.layers.convolution2d(inputs=self.conv2, num_outputs=64, kernel_size=[3, 3],
                                                         stride=[1, 1], padding='VALID', biases_initializer=None)
            self.conv4 = tf.contrib.layers.convolution2d(inputs=self.conv3, num_outputs=512, kernel_size=[7, 7],
                                                         stride=[1, 1], padding='VALID', biases_initializer=None)

            self.streamAC, self.streamVC = tf.split(3, 2, self.conv4)
            self.streamA = tf.contrib.layers.flatten(self.streamAC)
            self.streamV = tf.contrib.layers.flatten(self.streamVC)
            self.AW = tf.Variable(tf.random_normal([256, action_size]))
            self.VW = tf.Variable(tf.random_normal([256, 1]))
            self.Advantage = tf.matmul(self.streamA, self.AW)
            self.Value = tf.matmul(self.streamV, self.VW)
            self.outputLayer = self.Value + tf.subtract(self.Advantage,
                                                   tf.reduce_mean(self.Advantage, reduction_indices=1, keep_dims=True))
            self.maxOutputNode = tf.argmax(self.outputLayer, 1)


class ConvolutionalActorCriticNetwork:
    def __init__(self, scope='global', state_size=None, action_size=None):
        with tf.variable_scope(scope):
            self.inputs = tf.placeholder(shape=[None, state_size], dtype=tf.float32)
            # self.imageResize = tf.image.resize_images(self.inputs, [84, 84])
            self.imageIn = tf.reshape(self.inputs, shape=[-1, 84, 84, 1])
            self.conv1 = slim.conv2d(activation_fn=tf.nn.elu, inputs=self.imageIn, num_outputs=16, kernel_size=[8, 8],
                                     stride=[4, 4], padding='VALID')
            self.conv2 = slim.conv2d(activation_fn=tf.nn.elu, inputs=self.conv1, num_outputs=32, kernel_size=[4, 4],
                                     stride=[2, 2], padding='VALID')
            hidden = slim.fully_connected(slim.flatten(self.conv2), 256, activation_fn=tf.nn.elu)

            self.policyLayer = slim.fully_connected(hidden, action_size, activation_fn=tf.nn.softmax,
                                                    weights_initializer=normalized_columns_initializer(0.01),
                                                    biases_initializer=None)
            self.valueLayer = slim.fully_connected(hidden, 1, activation_fn=None,
                                                   weights_initializer=normalized_columns_initializer(1.0),
                                                   biases_initializer=None)
