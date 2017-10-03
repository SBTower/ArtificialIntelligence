"""Author: Stuart Tower
"""

import tensorflow as tf
import copy


# Deep Deterministic Policy Gradient
class DDPG():
    """An implementation of Deep Deterministic Policy Gradient

    This contains the current policy that converts the observed state into the optimal action, and the update rule for
    DDPG that update this policy based on experience. This algorithm is designed to handle continuous action spaces, and
    uses an Actor network and a Critic network.
    """
    def __init__(self, sess, actor_network, critic_network, learning_rate=0.1, discount_factor=0.99, target_network_update_rate=0.99):
        """

        :param sess: The top level tensorflow session to build the computation graph in
        :param actor_network: The network architecture for the actor network, which outputs which action to select
        :param critic_network: The network architecture for the critic network, which outputs the value of selecting a particular action
        :param learning_rate: The rate at which the network is updated
        :param discount_factor: The rate at which reward are diluted as they move further into the future
        :param target_network_update_rate: The rate at which the target network is moved towards the current network
        """
        self.sess = sess
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.target_network_update_rate = target_network_update_rate
        self.actor_network = actor_network
        self.target_actor_network = copy.copy(self.actor_network)
        self.target_actor_network.scope = 'target_' + self.actor_network.scope
        self.critic_network = critic_network
        self.target_critic_network = copy.copy(self.critic_network)
        self.target_critic_network.scope = 'target_' + self.critic_network.scope
        self.defineUpdateOperations()
        self.init = tf.global_variables_initializer()
        self.initialize_variables()

    def initialize_variables(self):
        """Initialise the parameters in the network

        :return: None
        """
        self.sess.run(self.init)

    def defineUpdateOperations(self):
        """Defines the operations that perform the update of the policy

        :return: None
        """
        # Calculate the gradients to apply to the actor network. The default 'minimize' function cannot be used as the
        # gradients are calculated from the critic network and applied to the actor network
        self.action_gradients = tf.placeholder(tf.float32, [None, self.actor_network.action_size])
        self.actor_gradients = tf.gradients(self.actor_network.policyLayer, self.actor_network.get_trainable_vars(),
                                            -self.action_gradients)
        self.optimizeActorNetwork = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(
            zip(self.actor_gradients, self.actor_network.get_trainable_vars()))

        # Operations to optimize the critic network
        self.measured_value = tf.placeholder(tf.float32, [None, 1])
        self.loss = tf.reduce_mean(tf.square(self.measured_value - self.critic_network.outputValue))
        self.optimizeCriticNetwork = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        # Calculate the gradients to feed into the actor network
        self.action_grads = tf.gradients(self.critic_network.outputValue, self.critic_network.action)

        # Update the target network for both the actor and the critic networks
        from_vars = self.actor_network.get_trainable_vars()
        to_vars = self.target_actor_network.get_trainable_vars()

        self.update_actor_target_ops = []
        for from_var, to_var in zip(from_vars, to_vars):
            self.update_actor_target_ops.append(to_var.assign(from_var * self.target_network_update_rate + to_var * (1 - self.target_network_update_rate)))

        from_vars = self.critic_network.get_trainable_vars()
        to_vars = self.target_critic_network.get_trainable_vars()

        self.update_critic_target_ops = []
        for from_var, to_var in zip(from_vars, to_vars):
            self.update_critic_target_ops.append(to_var.assign(from_var * self.target_network_update_rate + to_var * (1 - self.target_network_update_rate)))

    def update_target_actor_network(self):
        """Update the target actor network towards the current actor network

        :return: None
        """
        self.sess.run(self.update_actor_target_ops)

    def update_target_critic_network(self):
        """Update the target critic network towards the current critic network

        :return: None
        """
        self.sess.run(self.update_critic_target_ops)

    def update_target_network(self):
        """Update both the actor and the critic target networks

        :return: None
        """
        self.update_target_actor_network()
        self.update_target_critic_network()

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

        # Get the next action that would be chosen by the actor network
        next_action = self.sess.run(self.target_actor_network.policyLayer,
                                        feed_dict={self.target_actor_network.inputs: sample_next_state})
        # Get the estimate of the total future reward based on the current critic network and the chosen action
        value_estimate = self.sess.run(self.target_critic_network.outputValue,
                                       feed_dict={self.target_critic_network.inputs: sample_next_state,
                                                  self.target_critic_network.action: next_action})

        # Calculate the updated future reward estimate based on the actual action and reward
        for mem in range(len(sample_next_state)):
            if sample_did_finish[mem]:
                updated_future_reward_estimate.append([sample_rewards[mem]])
            else:
                updated_future_reward_estimate.append(sample_rewards[mem] + self.discount_factor * value_estimate[mem])
            state_samples.append(sample_current_state[mem])
            action_taken.append(sample_actions[mem])

        # Optimize the critic network based on the updated future reward estimates
        self.sess.run(self.optimizeCriticNetwork,
                      feed_dict={self.critic_network.inputs: state_samples, self.critic_network.action: action_taken,
                                 self.measured_value: updated_future_reward_estimate})

        # Get the actions chosen by the actor network for the input states
        a_outs = self.sess.run(self.actor_network.policyLayer, feed_dict={self.actor_network.inputs: state_samples})

        # Calculate the gradients to apply to the actor network from the critic network
        a_grads = self.sess.run(self.action_grads, feed_dict={self.critic_network.inputs: state_samples,
                                                              self.critic_network.action: a_outs})

        # Update the actor network using the calculated gradients
        self.sess.run(self.optimizeActorNetwork,
                      feed_dict={self.actor_network.inputs: state_samples, self.action_gradients: a_grads[0]})

    def get_highest_value_action(self, state):
        """Get the action that gives the highest total future reward based on the current policy

        Note that the output action is likely to be within a certain range (for example if the network uses a tanh
        activation function the output action will be between -1 and 1). Care should be taken to scale the output action
        to the appropriate value before applying it to the environment.
        :param state: The observed state of the environment
        :return: The action that maximises the expected total future reward
        """
        a = self.sess.run(self.actor_network.policyLayer, feed_dict={self.actor_network.inputs: [state]})
        return a[0]
