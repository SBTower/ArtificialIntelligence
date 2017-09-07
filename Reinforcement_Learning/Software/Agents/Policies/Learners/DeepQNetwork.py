import tensorflow as tf
import copy


class DeepQNetwork():
    def __init__(self, sess, network, learning_rate=0.1, discount_factor=0.99, tau=0.99):
        self.sess = sess
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.tau = tau
        self.network = network
        self.target_network = copy.copy(self.network)
        self.target_network.scope = 'target_' + self.network.scope
        self.defineUpdateOperations()
        self.init = tf.global_variables_initializer()
        self.initialize_variables()

    def initialize_variables(self):
        self.sess.run(self.init)

    def defineUpdateOperations(self):

        self.actionTaken = tf.placeholder(tf.int32, [None], name="actionTaken")
        self.actionMasks = tf.one_hot(self.actionTaken, self.network.action_size)
        self.estimated_action_value = tf.reduce_sum(tf.multiply(self.network.policyLayer, self.actionMasks),
                                                    reduction_indices=1)

        self.measured_action_value = tf.placeholder(tf.float32, [None, ])
        self.loss = tf.reduce_mean(tf.square(self.estimated_action_value - self.measured_action_value))
        tf.summary.scalar('loss', self.loss)

        local_vars = self.network.get_trainable_vars()
        self.updateModel = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, var_list=local_vars)

        from_vars = self.network.get_trainable_vars()
        to_vars = self.target_network.get_trainable_vars()

        self.update_target_ops = []
        for from_var, to_var in zip(from_vars, to_vars):
            self.update_target_ops.append(to_var.assign(from_var * self.tau + to_var * (1 - self.tau)))

    def update_target_network(self):
        self.sess.run(self.update_target_ops)

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

        allQ = self.sess.run(self.target_network.policyLayer, feed_dict={self.target_network.inputs: sample_next_state})

        for mem in range(len(sample_next_state)):
            if sample_did_finish[mem]:
                y_.append(sample_rewards[mem])
            else:
                maxQ = max(allQ[mem])
                y_.append(sample_rewards[mem] + self.discount_factor * maxQ)
            state_samples.append(sample_current_state[mem])
            action_taken.append(sample_actions[mem])

        self.sess.run(self.updateModel, feed_dict={self.network.inputs: state_samples, self.measured_action_value: y_,
                                                   self.actionTaken: action_taken})

    def get_highest_value_action(self, state):
        # pol = self.sess.run(self.network.policyLayer, feed_dict={self.network.inputs: [state]})
        a = self.sess.run(self.network.maxOutputNode, feed_dict={self.network.inputs: [state]})
        return a[0]
