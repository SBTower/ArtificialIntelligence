import csv
import numpy as np
import random
import tensorflow as tf
import os
from Learner import Learner
from Networks import ConvolutionalNetwork

class ConvNetValuePredictor(Learner):

  def buildNN(self):
    self.network = ConvolutionalNetwork(self.scope, self.stateSize, self.actionSize)
    self.targetNetwork = ConvolutionalNetwork('target_'+self.scope, self.stateSize, self.actionSize)

    self.actionTaken = tf.placeholder(tf.int32, [None], name="actionMasks")
    self.actionMasks = tf.one_hot(self.actionTaken, self.actionSize)
    self.estimated_action_value = tf.reduce_sum(tf.multiply(self.network.outputLayer, self.actionMasks), reduction_indices=1)
    self.measured_action_value = tf.placeholder(tf.float32, [None,])
    self.loss = tf.reduce_mean(tf.square(self.estimated_action_value - self.measured_action_value))
    self.updateModel = tf.train.AdamOptimizer(0.0001).minimize(self.loss)

    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'target_'+self.scope)

    self.update_target_ops = []
    for from_var,to_var in zip(from_vars, to_vars):
      self.update_target_ops.append(to_var.assign(from_var*self.tau + to_var*(1-self.tau)))

  def update(self, batch):
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

    allQ = self.sess.run(self.targetNetwork.outputLayer, feed_dict={self.targetNetwork.inputs:sampleNextState})

    for mem in range(len(sampleNextState)):
      if sampleDidFinish[mem]:
        y_.append(sampleRewards[mem])
      else:
        maxQ = max(allQ[mem])
        y_.append(sampleRewards[mem] + self.lmda*maxQ)
      state_samples.append(sampleCurrentState[mem])
      actionTaken.append(sampleActions[mem])
    
    self.sess.run(self.updateModel, feed_dict={self.network.inputs:state_samples, self.measured_action_value:y_, self.actionTaken:actionTaken})

  def updateTargetNetwork(self):    
    self.sess.run(self.update_target_ops)

  def getHighestValueAction(self, state):
    a = self.sess.run(self.network.maxOutputNode, feed_dict={self.network.inputs:[state]})
    return a[0]

