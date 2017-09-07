import numpy as np
import tensorflow as tf


class QLearning():
    def __init__(self, sess, network, learning_rate=0.1, discount_factor=0.99, tau=0.99):
        self.sess = sess
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.tau = tau
        self.network = network
        self.defineUpdateOperations()
        self.init = tf.global_variables_initializer()
        self.initialize_variables()

    def initialize_variables(self):
        self.sess.run(self.init)

    def defineUpdateOperations(self):
        self.updated_value = tf.placeholder(shape=[1, self.network.action_size], dtype=tf.float32)
        self.loss = tf.reduce_sum(tf.square(self.updated_value - self.network.outputLayer))
        self.trainer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)

        self.updateModel = self.trainer.minimize(self.loss)

    def update(self, batch):
        for experience in batch:
            estimated_future_value = self.sess.run(self.network.outputLayer,
                                                   feed_dict={self.network.inputs: [experience.next_state]})
            max_estimated_future_value = np.max(estimated_future_value)

            updated_action_value = self.sess.run(self.network.outputLayer,
                                                 feed_dict={self.network.inputs: [experience.state]})
            updated_action_value[0, experience.action] = experience.reward + self.discount_factor * max_estimated_future_value

            self.sess.run(self.updateModel, feed_dict={self.network.inputs: [experience.state],
                                                       self.updated_value: updated_action_value})

    def get_highest_value_action(self, state):
        a = self.sess.run(self.network.maxOutputNode, feed_dict={self.network.inputs: [state]})
        return a[0]
