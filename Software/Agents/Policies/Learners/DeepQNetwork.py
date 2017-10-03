"""Author: Stuart Tower
"""

import tensorflow as tf
import copy


class DeepQNetwork():
    """An implementation of Deep Q-Network: https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf

    This contains the current policy that converts the observed state into the optimal action, and the update rule for
    Deep Q-Networks that update this policy based on experience. This implementation makes use of a target network to
    stabilise the learning process.
    """
    def __init__(self, sess, network, learning_rate=0.1, discount_factor=0.99, target_network_update_rate=0.99):
        """Initialises the parameters for the learner

        :param sess: The top level tensorflow session to build the computation graph in
        :param network: The network architecture to use for the policy
        :param learning_rate: The rate at which the network is updated
        :param discount_factor: The rate at which reward are diluted as they move further into the future
        :param target_network_update_rate: The rate at which the target network is moved towards the current network
        """
        self.sess = sess
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.target_network_update_rate = target_network_update_rate
        self.network = network
        self.target_network = copy.copy(self.network)
        self.target_network.scope = 'target_' + self.network.scope
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
        # Create a one-hot representation of the action taken
        self.actionTaken = tf.placeholder(tf.int32, [None], name="actionTaken")
        self.actionMasks = tf.one_hot(self.actionTaken, self.network.action_size)

        # Calculate the estimated value of the action based on the current policy
        self.estimated_action_value = tf.reduce_sum(tf.multiply(self.network.policyLayer, self.actionMasks),
                                                    reduction_indices=1)

        # An input for the actual reward observed
        self.measured_action_value = tf.placeholder(tf.float32, [None, ])

        # Calculate the difference between the actual and expected rewards
        self.loss = tf.reduce_mean(tf.square(self.estimated_action_value - self.measured_action_value))

        # Update the variables in the active policy network, ensuring only the local variables are updated
        local_vars = self.network.get_trainable_vars()
        self.updateModel = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, var_list=local_vars)

        # Define the operation to update the target network towards the current policy
        from_vars = self.network.get_trainable_vars()
        to_vars = self.target_network.get_trainable_vars()
        self.update_target_ops = []
        for from_var, to_var in zip(from_vars, to_vars):
            self.update_target_ops.append(to_var.assign(from_var * self.target_network_update_rate + to_var * (1 - self.target_network_update_rate)))

    def update_target_network(self):
        """This function updates the target network towards the active network

        :return: None
        """
        self.sess.run(self.update_target_ops)

    def update(self, batch):
        """Updates the current policy based on the input batch of experiences

        :param batch: A collection of experiences which will be used to update the policy
        :return: None
        """
        # Define the lists to store the experiences. This is done to reduce the number of calls to the tensorflow session
        updated_future_reward_estimate = []
        state_samples = []
        action_taken = []

        sample_next_state = []
        sample_current_state = []
        sample_rewards = []
        sample_did_finish = []
        sample_actions = []

        # Convert the batch of experiences into separate lists of values
        for experience in batch:
            sample_next_state.append(experience.next_state)
            sample_current_state.append(experience.state)
            sample_rewards.append(experience.reward)
            sample_did_finish.append(experience.done)
            sample_actions.append(experience.action)

        # Get the previous future reward estimates from the policy for each state in the batch of experiences
        allQ = self.sess.run(self.target_network.policyLayer, feed_dict={self.target_network.inputs: sample_next_state})

        # Calculate the updated future reward estimate based on the actual reward and estimated future reward
        for mem in range(len(sample_next_state)):
            if sample_did_finish[mem]:
                updated_future_reward_estimate.append(sample_rewards[mem])
            else:
                maxQ = max(allQ[mem])
                updated_future_reward_estimate.append(sample_rewards[mem] + self.discount_factor * maxQ)
            state_samples.append(sample_current_state[mem])
            action_taken.append(sample_actions[mem])

        # Update the current policy towards the new estimated future reward based on the actual action and reward
        self.sess.run(self.updateModel, feed_dict={self.network.inputs: state_samples, self.measured_action_value: updated_future_reward_estimate,
                                                   self.actionTaken: action_taken})

    def get_highest_value_action(self, state):
        """Get the action that gives the highest total future reward based on the current policy

        :param state: The observed state of the environment
        :return: The action that maximises the expected total future reward
        """
        a = self.sess.run(self.network.maxOutputNode, feed_dict={self.network.inputs: [state]})
        return a[0]
