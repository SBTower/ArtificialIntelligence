"""Author: Stuart Tower
"""

import numpy as np
import tensorflow as tf


class QLearning():
    """An implementation of Q-Learning

    This contains the current policy that converts the observed state into the optimal action, and the update rule for
    Q-Learning that updates this policy based on experience.

    This implementation uses a neural network, but using a single layer network is equivalent to the tabular Q-Learning
    method.
    """
    def __init__(self, sess, network, learning_rate=0.1, discount_factor=0.99):
        """Initialise the parameters describing the learner

        :param sess: The top level tensorflow session to build the computation graph in
        :param network: The network architecture to use for the policy
        :param learning_rate: The rate at which the network is updated
        :param discount_factor: The rate at which reward are diluted as they move further into the future
        """
        self.sess = sess
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.network = network
        self.defineUpdateOperations()
        self.init = tf.global_variables_initializer()
        self.initialize_variables()

    def initialize_variables(self):
        """Initialises the variables in the computation graph in the current session

        :return: None
        """
        self.sess.run(self.init)

    def defineUpdateOperations(self):
        """Defines the operations that perform the update of the policy

        :return: None
        """
        self.updated_value = tf.placeholder(shape=[1, self.network.action_size], dtype=tf.float32)
        self.loss = tf.reduce_sum(tf.square(self.updated_value - self.network.policyLayer))
        self.trainer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)

        self.updateModel = self.trainer.minimize(self.loss)

    def update(self, batch):
        """Updates the current policy based on the input batch of experiences

        :param batch: A collection of experiences which will be used to update the policy
        :return: None
        """
        for experience in batch:
            # First calculate the expected future reward from the final state
            estimated_future_value = self.sess.run(self.network.policyLayer,
                                                   feed_dict={self.network.inputs: [experience.next_state]})
            max_estimated_future_value = np.max(estimated_future_value)

            # Get the estimated future reward for each action based on the initial state
            updated_action_value = self.sess.run(self.network.policyLayer,
                                                 feed_dict={self.network.inputs: [experience.state]})

            # Update the expected future reward for the selected action based on the actual reward obtained
            updated_action_value[0, experience.action] = experience.reward + self.discount_factor * max_estimated_future_value

            # Update the policy with the new estimated action values
            self.sess.run(self.updateModel, feed_dict={self.network.inputs: [experience.state],
                                                       self.updated_value: updated_action_value})

    def get_highest_value_action(self, state):
        """Get the action that gives the highest total future reward based on the current policy

        :param state: The observed state of the environment
        :return: The action that maximises the expected total future reward
        """
        a = self.sess.run(self.network.maxOutputNode, feed_dict={self.network.inputs: [state]})
        return a[0]
