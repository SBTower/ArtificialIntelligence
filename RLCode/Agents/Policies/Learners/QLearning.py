import numpy as np
import tensorflow as tf
from Learner import Learner

class QLearning(Learner):

  def buildNN(self):
    self.network = TabularNetwork(self.scope, self.stateSize, self.actionSize)

    self.updated_value = tf.placeholder(shape=[1,self.actionSize],dtype=tf.float32)
    self.loss = tf.reduce_sum(tf.square(self.updated_value - self.network.outputLayer))
    self.trainer = tf.train.GradientDescentOptimizer(learning_rate = self.alpha)

    if self.async is True:
      local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
      self.gradients = tf.gradients(self.loss,local_vars)
      grads, _ = tf.clip_by_global_norm(self.gradients,40.0)

      global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
      self.updateModel = trainer.apply_gradients(zip(grads, global_vars))

    else:
      self.updateModel = self.trainer.minimize(self.loss)

  def update(self, batch, ordered):
    for experience in batch:
      estimated_future_value = self.sess.run(self.network.outputLayer, feed_dict={self.network.inputs:[experience.nextState]})
      max_estimated_future_value = np.max(estimated_future_value)

      updated_action_value = self.sess.run(self.network.outputLayer, feed_dict={self.network.inputs:[experience.state]})
      updated_action_value[0, experience.action] = experience.reward + self.lmda * max_estimated_future_value

      self.sess.run(self.updateModel, feed_dict={self.network.inputs1:[experience.state], self.updated_value:updated_action_value})

  def getHighestValueAction(self, state):
    a = self.sess.run(self.network.maxOutputNode, feed_dict={self.network.inputs:[state]})
    return a[0]
