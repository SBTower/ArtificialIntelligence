import csv
import copy
import numpy as np
import random
import tensorflow as tf
import os
import scipy.signal
from Learner import Learner
from Networks import FullyConnectedNetwork, FullyConnectedCriticNetwork
# Deep Deterministic Policy Gradient
class DDPG(Learner):

  def buildNN(self):
    
    self.tau = 0.001

    self.actor_network = FullyConnectedNetwork('actor_'+self.scope, self.stateSize, self.actionSize)
    self.target_actor_network = FullyConnectedNetwork('actor_target_'+self.scope, self.stateSize, self.actionSize)

    self.action_gradients = tf.placeholder(tf.float32, [None, self.actionSize])

    self.actor_gradients = tf.gradients(self.actor_network.policyLayer, self.actor_network.getTrainableVars(), -self.action_gradients)

    self.optimizeActorNetwork = tf.train.AdamOptimizer(0.0001).apply_gradients(zip(self.actor_gradients, self.actor_network.getTrainableVars()))

    self.critic_network = FullyConnectedCriticNetwork('critic_'+self.scope, self.stateSize, self.actionSize)
    self.target_critic_network = FullyConnectedCriticNetwork('critic_target_'+self.scope, self.stateSize, self.actionSize)

    self.measured_value = tf.placeholder(tf.float32, [None,1])
    self.loss = tf.reduce_mean(tf.square(self.measured_value - self.critic_network.outputValue))
    self.optimizeCriticNetwork = tf.train.AdamOptimizer(0.001).minimize(self.loss)

    self.action_grads = tf.gradients(self.critic_network.outputValue, self.critic_network.action)

    from_vars = self.actor_network.getTrainableVars()
    to_vars = self.target_actor_network.getTrainableVars()

    self.update_actor_target_ops = []
    for from_var,to_var in zip(from_vars, to_vars):
      self.update_actor_target_ops.append(to_var.assign(from_var*self.tau + to_var*(1-self.tau)))

    from_vars = self.critic_network.getTrainableVars()
    to_vars = self.target_critic_network.getTrainableVars()

    self.update_critic_target_ops = []
    for from_var,to_var in zip(from_vars, to_vars):
      self.update_critic_target_ops.append(to_var.assign(from_var*self.tau + to_var*(1-self.tau)))

  def updateTargetActorNetwork(self):
    self.sess.run(self.update_actor_target_ops)

  def updateTargetCriticNetwork(self):
    self.sess.run(self.update_critic_target_ops)

  def updateTargetNetwork(self):
    self.updateTargetActorNetwork()
    self.updateTargetCriticNetwork()

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

    action_estimate = self.sess.run(self.target_actor_network.policyLayer, feed_dict={self.target_actor_network.inputs:sampleNextState}) 
    value_estimate = self.sess.run(self.target_critic_network.outputValue, feed_dict={self.target_critic_network.inputs:sampleNextState, self.target_critic_network.action:action_estimate})

    for mem in range(len(sampleNextState)):
      if sampleDidFinish[mem]:
        y_.append([sampleRewards[mem]])
      else:
        y_.append(sampleRewards[mem] + self.lmda*value_estimate[mem])
      state_samples.append(sampleCurrentState[mem])
      actionTaken.append(sampleActions[mem])

    self.sess.run(self.optimizeCriticNetwork, feed_dict={self.critic_network.inputs:state_samples, self.critic_network.action:actionTaken, self.measured_value:y_})

    a_outs = self.sess.run(self.actor_network.policyLayer, feed_dict={self.actor_network.inputs:state_samples})

    a_grads = self.sess.run(self.action_grads, feed_dict={self.critic_network.inputs:state_samples, self.critic_network.action:a_outs})

    self.sess.run(self.optimizeActorNetwork, feed_dict={self.actor_network.inputs:state_samples, self.action_gradients:a_grads[0]})

  def getHighestValueAction(self, state):
    a = self.sess.run(self.actor_network.policyLayer, feed_dict={self.actor_network.inputs:[state]})
    return a[0]

