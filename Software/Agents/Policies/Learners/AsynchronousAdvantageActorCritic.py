"""Author: Stuart Tower
NOTE - Requires testing and understanding
"""

import copy
import numpy as np
import tensorflow as tf
import scipy.signal


class AsynchronousAdvantageActorCritic():
    """An implementation of the Advantage Actor Critic algorithm

    This contains the current policy that converts the observed state into the optimal action, and the update rule for
    the advantage actor critic algorithm to update the policy based on experience. This method uses one network with two
    output layers to model the value and the advantage (similar to actor-critic methods)
    """
    def __init__(self, sess, network, learning_rate=0.1, discount_factor=0.99):
        """Initialises the parameters for the learner

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
        # Create a one-hot encoding of the action taken
        self.actionTaken = tf.placeholder(tf.int32, [None])
        self.actionMasks = tf.one_hot(self.actionTaken, self.network.action_size)

        # Calculate the estimated value of the action based on the current policy
        self.estimated_action_value = tf.reduce_sum(self.network.policyLayer * self.actionMasks, [1])

        # Input placeholders for the advantage and target values
        self.target_value = tf.placeholder(tf.float32, [None])
        self.advantages = tf.placeholder(tf.float32, [None])

        # Calculate the loss and advantage
        self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_value - tf.reshape(self.network.valueLayer, [-1])))
        self.entropy = - tf.reduce_sum(self.network.policyLayer * tf.log(self.network.policyLayer))
        self.policy_loss = -tf.reduce_sum(tf.log(self.estimated_action_value) * self.advantages)
        self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * 0.01

        # Create operations that update the global graph with the values in the local graph
        local_vars = self.network.get_trainable_vars()
        self.gradients = tf.gradients(self.loss, local_vars)
        grads, _ = tf.clip_by_global_norm(self.gradients, 40.0)

        global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
        self.updateModel = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(zip(grads, global_vars))

        self.update_network_op_holder = []
        for global_var, local_var in zip(global_vars, local_vars):
            self.update_network_op_holder.append(local_var.assign(global_var))


    def initialise_variables(self):
        """Initialises the variables in the computation graph in the current session

        :return: None
        """
        self.sess.run(self.init)
        self.sess.run(self.update_network_op_holder)

    def update(self, batch):
        """Updates the current policy based on the input batch of experiences

        :param batch: A collection of experiences which will be used to update the policy. Note that this assumes an
        ordered batch of experiences, where each experience directly follows the previous one
        :return: None
        """

        # Define the lists to store the experiences. This is done to reduce the number of calls to the tensorflow session
        sampleNextState = []
        sampleCurrentState = []
        sampleRewards = []
        sampleDidFinish = []
        sampleActions = []

        # Convert the batch of experiences into separate lists of values
        for experience in batch:
            sampleNextState.append(experience.next_state)
            sampleCurrentState.append(experience.state)
            sampleRewards.append(experience.reward)
            sampleDidFinish.append(experience.done)
            sampleActions.append(experience.action)

        # Get the previous future reward estimates from the policy for each state in the batch of experiences
        value_estimate = self.sess.run(self.network.valueLayer,
                                       feed_dict={self.network.inputs: [sampleNextState[len(sampleNextState) - 1]]})[0, 0]
        # Calculate the discounted rewards for each experience in the batch
        rewards_plus = copy.copy(sampleRewards)
        rewards_plus.append(value_estimate)
        rewards_plus = np.asarray(rewards_plus)
        discounted_rewards = scipy.signal.lfilter([1], [1, -0.99], rewards_plus[::-1], axis=0)[::-1][:-1]

        # Calculate the discounted advantage of each state in the batch of experiences
        values = self.sess.run(self.network.valueLayer, feed_dict={self.network.inputs: sampleCurrentState})
        values = [v[0] for v in values]
        values.append(value_estimate)
        values = np.asarray(values)
        sampleRewards = np.asarray(sampleRewards)
        advantages = sampleRewards + 0.99 * values[1:] - values[:-1]
        discounted_advantages = scipy.signal.lfilter([1], [1, -0.99], advantages[::-1], axis=0)[::-1]

        # Update the model based on the experience
        self.sess.run(self.updateModel,
                      feed_dict={self.network.inputs: sampleCurrentState, self.target_value: discounted_rewards,
                                 self.advantages: discounted_advantages, self.actionTaken: sampleActions})

        # Update the global network
        self.sess.run(self.update_network_op_holder)

    def get_highest_value_action(self, state):
        """Get the action that gives the highest total future reward based on the current policy

        :param state: The observed state of the environment
        :return: The action that maximises the expected total future reward
        """
        a = self.sess.run(self.network.maxOutputNode, feed_dict={self.network.inputs:[state]})
        # a_dist = self.sess.run(self.network.policyLayer, feed_dict={self.network.inputs: [state]})
        # a = np.random.choice(a_dist[0], p=a_dist[0])
        # a = np.argmax(a_dist == a)
        return a[0]
