import numpy as np
import tensorflow as tf
import os

class Learner:

  def __init__(self, sess, scope='worker_0', stateSize = 0, actionSize = 0, alpha = 0.1, lmda = 0.99, async = False):
    if type(stateSize) is tuple:
      self.stateSize = stateSize[0]
    else:
      self.stateSize = stateSize
    self.actionSize = actionSize
    self.alpha = alpha
    self.lmda = lmda
    self.sess = sess
    self.async = async
    self.scope = scope
    self.buildNN()
    self.init = tf.global_variables_initializer()
    self.initialiseVariables()
    self.saver = tf.train.Saver(max_to_keep = 100)

  def buildNN(self):
    pass

  def initialiseVariables(self):
    self.sess.run(self.init)

  def update(self, batch, ordered):
    pass

  def getHighestValueAction(self, state):
    pass

  def saveNetwork(self, filename):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path += '/Models/'
    dir_path += filename
    self.saver.save(self.sess, dir_path)

  def loadNetwork(self, filename):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path += '/Models/'
    dir_path += filename
    self.saver.restore(self.sess, dir_path)
