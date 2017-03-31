import tensorflow as tf
import numpy as np
import tflearn
import tensorflow.contrib.slim as slim

def normalized_columns_initializer(std=1.0):
  def _initializer(shape, dtype=None, partition_info=None):
    out = np.random.randn(*shape).astype(np.float32)
    out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
    return tf.constant(out)
  return _initializer

class FullyConnectedNetwork:

  def __init__(self, scope = 'global', stateSize = None, actionSize = None):
    self.scope = scope
    with tf.variable_scope(scope):
      self.inputs = tf.placeholder(shape=[None,stateSize],dtype=tf.float32)
      self.layer1 = slim.fully_connected(self.inputs, 400, activation_fn = tf.nn.relu)
      self.layer2 = slim.fully_connected(self.layer1, 300, activation_fn = tf.nn.relu)

      self.policyLayer = slim.fully_connected(self.layer2, actionSize, activation_fn=tf.nn.tanh, weights_initializer=normalized_columns_initializer(0.3))

      self.maxOutputNode = tf.argmax(self.policyLayer, 1)

  def getTrainableVars(self):
    params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
    return params

class FullyConnectedDuelingNetwork:

  def __init__(self, scope = 'global', stateSize = None, actionSize = None):
    self.scope = scope
    with tf.variable_scope(scope):
      self.inputs = tf.placeholder(shape=[None,stateSize],dtype=tf.float32)
      self.W1 = tf.Variable(tf.random_uniform([stateSize, 400],-.1,.1))
      self.b1 = tf.Variable(tf.random_uniform([400],-.1,.1))
      self.layer1 = tf.nn.relu(tf.matmul(self.inputs, self.W1) + self.b1)
      self.W2 = tf.Variable(tf.random_uniform([400, 300],-.1,.1))
      self.b2 = tf.Variable(tf.random_uniform([300],-.1,.1))
      self.layer2 = tf.nn.relu(tf.matmul(self.layer1, self.W2) + self.b2)
      #self.W3 = tf.Variable(tf.random_uniform([600, 300],-.01,.01))
      #self.b3 = tf.Variable(tf.random_uniform([300],-.01,.01))
      #self.layer3 = tf.matmul(self.layer2, self.W3) + self.b3

      self.valueWeight = tf.Variable(tf.random_uniform([300, 1],-.1,.1))
      self.valueBias = tf.Variable(tf.random_uniform([1],-.1,.1))
      self.valueLayer = tf.matmul(self.layer2, self.valueWeight) + self.valueBias

      self.advantageWeight = tf.Variable(tf.random_uniform([300, actionSize],-.1,.1))
      self.advantageBias = tf.Variable(tf.random_uniform([actionSize],-.1,.1))
      self.advantageLayer = tf.matmul(self.layer2, self.advantageWeight) + self.advantageBias

      self.policyLayer = self.valueLayer + tf.subtract(self.advantageLayer, tf.reduce_mean(self.advantageLayer, reduction_indices=1, keep_dims=True))

      self.maxOutputNode = tf.argmax(self.policyLayer, 1)

  def getTrainableVars(self):
    params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
    return params

class FullyConnectedActorCriticNetwork:

  def __init__(self, scope = 'global', stateSize = None, actionSize = None):
    self.scope = scope
    with tf.variable_scope(scope):
      self.inputs = tf.placeholder(shape=[None,stateSize],dtype=tf.float32)
      self.W1 = tf.Variable(tf.random_uniform([stateSize, 400],-.01,.01))
      self.b1 = tf.Variable(tf.random_uniform([400],-.01,.01))
      self.layer1 = tf.nn.relu(tf.matmul(self.inputs, self.W1) + self.b1)
      self.W2 = tf.Variable(tf.random_uniform([400, 300],-.01,.01))
      self.b2 = tf.Variable(tf.random_uniform([300],-.01,.01))
      self.layer2 = tf.nn.relu(tf.matmul(self.layer1, self.W2) + self.b2)
      #self.W3 = tf.Variable(tf.random_uniform([600, 400],-.01,.01))
      #self.b3 = tf.Variable(tf.random_uniform([400],-.01,.01))
      #self.layer3 = tf.nn.relu(tf.matmul(self.layer2, self.W3) + self.b3)

      self.policyLayer = slim.fully_connected(self.layer2, actionSize, activation_fn=tf.nn.softmax, weights_initializer=normalized_columns_initializer(0.003), biases_initializer = None)

      self.valueLayer = slim.fully_connected(self.layer2, 1, activation_fn=None, weights_initializer=normalized_columns_initializer(1.0), biases_initializer=None)

      self.maxOutputNode = tf.argmax(self.policyLayer, 1)

  def getTrainableVars(self):
    params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
    return params

class FullyConnectedCriticNetwork:

  def __init__(self, scope = 'global', stateSize = None, actionSize = None):
    self.scope = scope
    with tf.variable_scope(scope):
      self.inputs = tflearn.input_data(shape=[None, stateSize])
      self.action = tflearn.input_data(shape=[None, actionSize])
      self.layer1 = tflearn.fully_connected(self.inputs, 400, activation='relu')
      self.t1 = tflearn.fully_connected(self.layer1, 300)
      self.t2 = tflearn.fully_connected(self.action, 300)
      self.layer2 = tflearn.activation(tf.matmul(self.layer1, self.t1.W) + tf.matmul(self.action, self.t2.W) + self.t2.b, activation='relu')
      weight_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
      self.outputValue = tflearn.fully_connected(self.layer2, 1, weights_init = weight_init)


#      self.inputs = tf.placeholder(shape=[None,stateSize],dtype=tf.float32)
#      self.action = tf.placeholder(shape=[None,actionSize],dtype=tf.float32)
#      self.layer1 = slim.fully_connected(self.inputs, 400, activation_fn=tf.nn.relu)
#      self.mid = tf.concat(1,[self.layer1,self.action])
#      self.outputValue = slim.fully_connected(self.mid, 1, activation_fn = tf.nn.relu)

  def getTrainableVars(self):
    params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
    return params










