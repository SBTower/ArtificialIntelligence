"""Author: Stuart Tower
"""

import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import os


class NetworkBase:
    """A base network class to be extended
    """
    def __init__(self, scope='global'):
        self.scope = scope

    def get_trainable_vars(self):
        """Get the variables that are to be trained

        :return: The trainable parameters
        """
        params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
        return params

    def save_network(self, sess, filename):
        """Save the model to a file.
        UPDATE TO USE TENSORFLOW 'SAVEDMODEL'

        :param sess: The tensorflow session to use
        :param filename: The name of the file to save
        :return: None
        """
        dir_path = os.path.dirname(os.path.realpath(__file__))
        dir_path += '/Models/'
        dir_path += filename
        saver = tf.train.Saver()
        saver.save(sess, dir_path)

    def load_network(self, sess, filename):
        """Load a model from a file
        UPDATE TO USE TENSORFLOW 'SAVEDMODEL'

        :param sess: The tensorflow session to use
        :param filename: The name of the file to load
        :return: None
        """
        dir_path = os.path.dirname(os.path.realpath(__file__))
        dir_path += '/Models/'
        dir_path += filename
        saver = tf.train.Saver()
        saver.restore(sess, dir_path)

class FullyConnectedNetwork(NetworkBase):
    """A simple fully connected neural network
    """
    def __init__(self, state_size, action_size, scope='global', layer_size=np.array([400, 300])):
        """Build the neural network architecture

        :param state_size: The size of the state space, used as the size of the input layer
        :param action_size: The size of the action space, used as the size of the output layer
        :param scope: The scope to assign to the model in tensorflow
        :param layer_size: The size of the hidden layers in the network
        """
        self.state_size = state_size
        self.action_size = action_size
        self.scope = scope
        with tf.variable_scope(scope):
            self.inputs = tf.placeholder(shape=[None, state_size], dtype=tf.float32)
            self.layers = [self.inputs]
            for i in range(len(layer_size)):
               self.layers.append(slim.fully_connected(self.layers[i], int(layer_size[i]), activation_fn=tf.nn.relu))

            self.policyLayer = slim.fully_connected(self.layers[-1], action_size, activation_fn=tf.nn.tanh)
            # Get the index of the highest output from the neural network
            self.maxOutputNode = tf.argmax(self.policyLayer, 1)



class FullyConnectedDuelingNetwork(NetworkBase):
    """A fully connected network using the dueling architecture
    """
    def __init__(self, state_size, action_size, scope='global', layer_size=np.array([400, 300])):
        """Build the dueling neural network architecture

        :param state_size: The size of the state space, used as the size of the input layer
        :param action_size: The size of the action space, used as the size of the output layer
        :param scope: The scope to assign to the model in tensorflow
        :param layer_size: The size of the hidden layers in the network
        """
        self.state_size = state_size
        self.action_size = action_size
        self.scope = scope
        with tf.variable_scope(scope):
            # Build the input layer
            self.inputs = tf.placeholder(shape=np.concatenate([[None], state_size]).tolist(), dtype=tf.float32)
            self.layers = [self.inputs]
            # Build the hidden layers
            for i in range(len(layer_size)):
                self.layers.append(slim.fully_connected(self.layers[i], int(layer_size[i]), activation_fn=tf.nn.relu))

            # Split the layer into a value and advantage layer
            self.valueWeight = tf.Variable(tf.random_uniform([int(layer_size[-1]), 1], -.1, .1))
            self.valueBias = tf.Variable(tf.random_uniform([1], -.1, .1))
            self.valueLayer = tf.matmul(self.layers[-1], self.valueWeight) + self.valueBias

            self.advantageWeight = tf.Variable(tf.random_uniform([int(layer_size[-1]), action_size], -.1, .1))
            self.advantageBias = tf.Variable(tf.random_uniform([action_size], -.1, .1))
            self.advantageLayer = tf.matmul(self.layers[-1], self.advantageWeight) + self.advantageBias

            # Combine the value and advantage layers into a single output layer
            self.policyLayer = self.valueLayer + tf.subtract(self.advantageLayer,
                                                             tf.reduce_mean(self.advantageLayer, reduction_indices=1,
                                                                            keep_dims=True))
            # get the index of the highest output from the neural network
            self.maxOutputNode = tf.argmax(self.policyLayer, 1)


class FullyConnectedActorCriticNetwork(NetworkBase):
    """A fully connected actor-critic network
    """
    def __init__(self, state_size, action_size, scope='global', layer_size = np.array([50])):
        """

        :param state_size: The size of the state space, used as the size of the input layer
        :param action_size: The size of the action space, used as the size of the output layer
        :param scope: The scope to assign to the model in tensorflow
        :param layer_size: The size of the hidden layers in the network
        """
        self.state_size = state_size
        self.action_size = action_size
        self.scope = scope
        with tf.variable_scope(scope):
            # Build the input layer
            self.inputs = tf.placeholder(shape=np.concatenate([[None], state_size]).tolist(), dtype=tf.float32)
            self.layers = [self.inputs]
            # Builde the hidden layers
            for i in range(len(layer_size)):
                self.layers.append(slim.fully_connected(self.layers[i], int(layer_size[i]), activation_fn=tf.nn.relu))

            # The policy layer acts as the 'actor'
            self.policyLayer = slim.fully_connected(self.layers[-1], action_size, activation_fn=tf.nn.softmax,
                                                    biases_initializer=None)

            # The value layer acts as the 'critic'
            self.valueLayer = slim.fully_connected(self.layers[-1], 1, activation_fn=None,
                                                   biases_initializer=None)
            # Get the index of the highest value output from the policy layer
            self.maxOutputNode = tf.argmax(self.policyLayer, 1)


class FullyConnectedCriticNetwork(NetworkBase):
    """A fully connected critic network, which takes both state and action as input and outputs value
    """
    def __init__(self, state_size, action_size, scope='global'):
        """

        :param state_size: The size of the state space, used as the size of the input layer
        :param action_size: The size of the action space, used as the size of the output layer
        :param scope: The scope to assign to the model in tensorflow
        """
        self.state_size = state_size
        self.action_size = action_size
        self.scope = scope
        with tf.variable_scope(scope):
            # Build the two input layers
            self.inputs = tf.placeholder(shape=[None, state_size], dtype=tf.float32)
            self.action = tf.placeholder(shape=[None, action_size], dtype=tf.float32)

            self.layer1 = slim.fully_connected(self.inputs, 400, activation_fn=tf.nn.relu)
            self.W2 = tf.Variable(tf.random_uniform([400, 300], -.01, .01))
            self.W2_actions = tf.Variable(tf.random_uniform([action_size, 300], -.01, .01))
            self.b2 = tf.Variable(tf.random_uniform([300], -.01, .01))
            # Combine the two inputs to a single layer (possibly a better way of doing this with tf.flatten?)
            self.layer2 = tf.nn.relu(
                tf.matmul(self.layer1, self.W2) + tf.matmul(self.action, self.W2_actions) + self.b2)
            # Build the output layer, representing the value of a state-action pair
            self.outputValue = slim.fully_connected(self.layer2, 1, activation_fn=tf.nn.relu)
