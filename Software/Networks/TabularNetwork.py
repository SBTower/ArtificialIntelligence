"""Author: Stuart Tower
"""

import tensorflow as tf
import os


class TabularNetwork:
    """A Neural Network that is analogous to a table mapping state-actions to rewards
    """
    def __init__(self, state_size, action_size, scope='global'):
        """Initialise the network

        :param state_size: The size of the state space, used as the size of the input layer
        :param action_size: The size of the action space, used as the size of the output layer
        :param scope: The scope to assign to the network architecture
        """
        self.state_size = state_size
        self.action_size = action_size
        with tf.variable_scope(scope):
            self.inputs = tf.placeholder(shape=[1, state_size], dtype=tf.float32)
            self.W = tf.Variable(tf.random_uniform([state_size, action_size], 0, 0.01))
            self.policyLayer = tf.matmul(self.inputs, self.W)
            self.maxOutputNode = tf.argmax(self.policyLayer, 1)

    def save_network(self, sess, filename):
        """Save the network

        :param sess: The tensorflow session to save
        :param filename: The name of the file to save
        :return: None
        """
        dir_path = os.path.dirname(os.path.realpath(__file__))
        dir_path += '/Models/'
        dir_path += filename
        saver = tf.train.Saver()
        saver.save(sess, dir_path)

    def load_network(self, sess, filename):
        """Load a previously saved network. The network must be initialised with the same architecture before being
        loaded. ONLY THE WEIGHTS ARE SAVED AND LOADED
        UPDATE THIS TO USE 'SAVEDMODEL' FROM TENSORFLOW

        :param sess: The tensorflow session to load
        :param filename: The name of the file to load
        :return: None
        """
        dir_path = os.path.dirname(os.path.realpath(__file__))
        dir_path += '/Models/'
        dir_path += filename
        saver = tf.train.Saver()
        saver.restore(sess, dir_path)