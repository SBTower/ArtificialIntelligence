import csv
import copy
import numpy as np
import random
import tensorflow as tf
import os
import scipy.signal
from Learner import Learner
from Networks import ConvolutionalActorCriticNetwork, FullyConnectedActorCriticNetwork

class AdvantageActorCritic(Learner):

  def buildNN(self):

    self.network = FullyConnectedActorCriticNetwork(self.scope, self.stateSize, self.actionSize)

    self.actionTaken = tf.placeholder(tf.int32, [None])
    self.actionMasks = tf.one_hot(self.actionTaken, self.actionSize)

    self.responsible_outputs = tf.reduce_sum(self.network.policyLayer * self.actionMasks, [1])

    self.target_value = tf.placeholder(tf.float32, [None])
    self.advantages = tf.placeholder(tf.float32, [None])

    self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_value - tf.reshape(self.network.valueLayer, [-1])))
    self.entropy = - tf.reduce_sum(self.network.policyLayer * tf.log(self.network.policyLayer))
    self.policy_loss = -tf.reduce_sum(tf.log(self.responsible_outputs)*self.advantages)
    self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * 0.01

    if self.async is True:
      local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
      self.gradients = tf.gradients(self.loss,local_vars)
      grads, _ = tf.clip_by_global_norm(self.gradients,40.0)

      global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
      self.updateModel = tf.train.AdamOptimizer(0.0001).apply_gradients(zip(grads, global_vars))

      self.update_network_op_holder = []
      for global_var, local_var in zip(global_vars, local_vars):
        self.update_network_op_holder.append(local_var.assign(global_var))

    else:
      self.updateModel = tf.train.AdamOptimizer(0.0001).minimize(self.loss)
      self.update_network_op_holder = []

  def update(self, batch, ordered):
    y_ = []
    state_samples = []
    actionTaken = []

    sampleNextState = []
    sampleCurrentState = []
    sampleRewards = []
    sampleDidFinish = []
    sampleActions = []
    
    for experience in batch:
      sampleNextState.append(experience.nextState)
      sampleCurrentState.append(experience.state)
      sampleRewards.append(experience.reward)
      sampleDidFinish.append(experience.done)
      sampleActions.append(experience.action)

    value_estimate = self.sess.run(self.network.valueLayer, feed_dict={self.network.inputs:[sampleNextState[len(sampleNextState)-1]]})[0,0]
    rewards_plus = copy.copy(sampleRewards)
    rewards_plus.append(value_estimate)
    rewards_plus = np.asarray(rewards_plus)
    discounted_rewards = scipy.signal.lfilter([1], [1, -0.99], rewards_plus[::-1], axis=0)[::-1][:-1]
 
    values = self.sess.run(self.network.valueLayer, feed_dict={self.network.inputs:sampleCurrentState})
    values = [v[0] for v in values]
    values.append(value_estimate)
    values = np.asarray(values)
    sampleRewards = np.asarray(sampleRewards)
    advantages = sampleRewards + 0.99 * values[1:] - values[:-1]
    discounted_advantages = scipy.signal.lfilter([1], [1, -0.99], advantages[::-1], axis=0)[::-1]

    self.sess.run(self.updateModel, feed_dict={self.network.inputs:sampleCurrentState, self.target_value:discounted_rewards, self.advantages:discounted_advantages, self.actionTaken:sampleActions})

    self.sess.run(self.update_network_op_holder)

  def getHighestValueAction(self, state):
    #a = self.sess.run(self.network.maxOutputNode, feed_dict={self.network.inputs:[state]})
    a_dist = self.sess.run(self.network.policyLayer, feed_dict={self.network.inputs:[state]})
    a = np.random.choice(a_dist[0],p=a_dist[0])
    a = np.argmax(a_dist == a)
    return a

