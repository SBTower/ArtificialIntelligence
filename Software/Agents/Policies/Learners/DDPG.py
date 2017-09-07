import tensorflow as tf
import copy


# Deep Deterministic Policy Gradient
class DDPG():
    def __init__(self, sess, actor_network, critic_network, learning_rate=0.1, discount_factor=0.99, tau=0.99):
        self.sess = sess
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.tau = tau
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
        self.sess.run(self.init)

    def defineUpdateOperations(self):
        self.action_gradients = tf.placeholder(tf.float32, [None, self.actor_network.action_size])

        self.actor_gradients = tf.gradients(self.actor_network.policyLayer, self.actor_network.get_trainable_vars(),
                                            -self.action_gradients)

        self.optimizeActorNetwork = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(
            zip(self.actor_gradients, self.actor_network.get_trainable_vars()))

        self.measured_value = tf.placeholder(tf.float32, [None, 1])
        self.loss = tf.reduce_mean(tf.square(self.measured_value - self.critic_network.outputValue))
        self.optimizeCriticNetwork = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        self.action_grads = tf.gradients(self.critic_network.outputValue, self.critic_network.action)

        from_vars = self.actor_network.get_trainable_vars()
        to_vars = self.target_actor_network.get_trainable_vars()

        self.update_actor_target_ops = []
        for from_var, to_var in zip(from_vars, to_vars):
            self.update_actor_target_ops.append(to_var.assign(from_var * self.tau + to_var * (1 - self.tau)))

        from_vars = self.critic_network.get_trainable_vars()
        to_vars = self.target_critic_network.get_trainable_vars()

        self.update_critic_target_ops = []
        for from_var, to_var in zip(from_vars, to_vars):
            self.update_critic_target_ops.append(to_var.assign(from_var * self.tau + to_var * (1 - self.tau)))

    def update_target_actor_network(self):
        self.sess.run(self.update_actor_target_ops)

    def update_target_critic_network(self):
        self.sess.run(self.update_critic_target_ops)

    def update_target_network(self):
        self.update_target_actor_network()
        self.update_target_critic_network()

    def update(self, batch):
        y_ = []
        state_samples = []
        action_taken = []

        sample_next_state = []
        sample_current_state = []
        sample_rewards = []
        sample_did_finish = []
        sample_actions = []

        for experience in batch:
            sample_next_state.append(experience.next_state)
            sample_current_state.append(experience.state)
            sample_rewards.append(experience.reward)
            sample_did_finish.append(experience.done)
            sample_actions.append(experience.action)

        action_estimate = self.sess.run(self.target_actor_network.policyLayer,
                                        feed_dict={self.target_actor_network.inputs: sample_next_state})
        value_estimate = self.sess.run(self.target_critic_network.outputValue,
                                       feed_dict={self.target_critic_network.inputs: sample_next_state,
                                                  self.target_critic_network.action: action_estimate})

        for mem in range(len(sample_next_state)):
            if sample_did_finish[mem]:
                y_.append([sample_rewards[mem]])
            else:
                y_.append(sample_rewards[mem] + self.discount_factor * value_estimate[mem])
            state_samples.append(sample_current_state[mem])
            action_taken.append(sample_actions[mem])

        self.sess.run(self.optimizeCriticNetwork,
                      feed_dict={self.critic_network.inputs: state_samples, self.critic_network.action: action_taken,
                                 self.measured_value: y_})

        a_outs = self.sess.run(self.actor_network.policyLayer, feed_dict={self.actor_network.inputs: state_samples})

        a_grads = self.sess.run(self.action_grads, feed_dict={self.critic_network.inputs: state_samples,
                                                              self.critic_network.action: a_outs})

        self.sess.run(self.optimizeActorNetwork,
                      feed_dict={self.actor_network.inputs: state_samples, self.action_gradients: a_grads[0]})

    def get_highest_value_action(self, state):
        a = self.sess.run(self.actor_network.policyLayer, feed_dict={self.actor_network.inputs: [state]})
        return a[0]*2
