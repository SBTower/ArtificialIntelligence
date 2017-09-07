import tensorflow as tf
import os


class TabularNetwork:
    def __init__(self, state_size, action_size, scope='global'):
        self.state_size = state_size
        self.action_size = action_size
        with tf.variable_scope(scope):
            self.inputs = tf.placeholder(shape=[1, state_size], dtype=tf.float32)
            self.W = tf.Variable(tf.random_uniform([state_size, action_size], 0, 0.01))
            self.outputLayer = tf.matmul(self.inputs, self.W)
            self.maxOutputNode = tf.argmax(self.outputLayer, 1)

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