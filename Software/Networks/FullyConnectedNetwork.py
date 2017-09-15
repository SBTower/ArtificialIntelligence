import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import os


class NetworkBase:
    def __init__(self, scope='global'):
        self.scope = scope

    def get_trainable_vars(self):
        params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
        return params

    def save_network(self, sess, filename):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        dir_path += '/Models/'
        dir_path += filename
        saver = tf.train.Saver()
        saver.save(sess, dir_path)

    def load_network(self, sess, filename):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        dir_path += '/Models/'
        dir_path += filename
        saver = tf.train.Saver()
        saver.restore(sess, dir_path)

class FullyConnectedNetwork(NetworkBase):
    def __init__(self, state_size, action_size, scope='global', layer_size=np.array([400, 300])):
        self.state_size = state_size
        self.action_size = action_size
        self.scope = scope
        with tf.variable_scope(scope):
            self.inputs = tf.placeholder(shape=[None, state_size], dtype=tf.float32)
            self.layers = [self.inputs]
            for i in range(len(layer_size)):
               self.layers.append(slim.fully_connected(self.layers[i], int(layer_size[i]), activation_fn=tf.nn.relu))

            self.policyLayer = slim.fully_connected(self.layers[-1], action_size, activation_fn=tf.nn.tanh)

            self.maxOutputNode = tf.argmax(self.policyLayer, 1)



class FullyConnectedDuelingNetwork(NetworkBase):
    def __init__(self, state_size, action_size, scope='global', layer_size=np.array([400, 300])):
        self.state_size = state_size
        self.action_size = action_size
        self.scope = scope
        with tf.variable_scope(scope):
            self.inputs = tf.placeholder(shape=np.concatenate([[None], state_size]).tolist(), dtype=tf.float32)
            self.layers = [self.inputs]
            for i in range(len(layer_size)):
                self.layers.append(slim.fully_connected(self.layers[i], int(layer_size[i]), activation_fn=tf.nn.relu))

            self.valueWeight = tf.Variable(tf.random_uniform([int(layer_size[-1]), 1], -.1, .1))
            self.valueBias = tf.Variable(tf.random_uniform([1], -.1, .1))
            self.valueLayer = tf.matmul(self.layers[-1], self.valueWeight) + self.valueBias

            self.advantageWeight = tf.Variable(tf.random_uniform([int(layer_size[-1]), action_size], -.1, .1))
            self.advantageBias = tf.Variable(tf.random_uniform([action_size], -.1, .1))
            self.advantageLayer = tf.matmul(self.layers[-1], self.advantageWeight) + self.advantageBias

            self.policyLayer = self.valueLayer + tf.subtract(self.advantageLayer,
                                                             tf.reduce_mean(self.advantageLayer, reduction_indices=1,
                                                                            keep_dims=True))

            self.maxOutputNode = tf.argmax(self.policyLayer, 1)


class FullyConnectedActorCriticNetwork(NetworkBase):
    def __init__(self, state_size, action_size, scope='global', layer_size = np.array([50])):
        self.state_size = state_size
        self.action_size = action_size
        self.scope = scope
        with tf.variable_scope(scope):
            self.inputs = tf.placeholder(shape=np.concatenate([[None], state_size]).tolist(), dtype=tf.float32)
            self.layers = [self.inputs]
            for i in range(len(layer_size)):
                self.layers.append(slim.fully_connected(self.layers[i], int(layer_size[i]), activation_fn=tf.nn.relu))

            self.policyLayer = slim.fully_connected(self.layers[-1], action_size, activation_fn=tf.nn.softmax,
                                                    biases_initializer=None)

            self.valueLayer = slim.fully_connected(self.layers[-1], 1, activation_fn=None,
                                                   biases_initializer=None)

            self.maxOutputNode = tf.argmax(self.policyLayer, 1)


class FullyConnectedCriticNetwork(NetworkBase):
    def __init__(self, state_size, action_size, scope='global'):
        self.state_size = state_size
        self.action_size = action_size
        self.scope = scope
        with tf.variable_scope(scope):
            self.inputs = tf.placeholder(shape=[None, state_size], dtype=tf.float32)
            self.action = tf.placeholder(shape=[None, action_size], dtype=tf.float32)

            self.layer1 = slim.fully_connected(self.inputs, 400, activation_fn=tf.nn.relu)
            self.W2 = tf.Variable(tf.random_uniform([400, 300], -.01, .01))
            self.W2_actions = tf.Variable(tf.random_uniform([action_size, 300], -.01, .01))
            self.b2 = tf.Variable(tf.random_uniform([300], -.01, .01))
            self.layer2 = tf.nn.relu(
                tf.matmul(self.layer1, self.W2) + tf.matmul(self.action, self.W2_actions) + self.b2)
            self.outputValue = slim.fully_connected(self.layer2, 1, activation_fn=tf.nn.relu)
