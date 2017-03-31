import tensorflow as tf

class TabularNetwork:

  def __init__(self, scope = 'global', stateSize = None, actionSize = None):
    with tf.variable_scope(scope):
      self.inputs1 = tf.placeholder(shape=[1,stateSize],dtype=tf.float32)
      self.W = tf.Variable(tf.random_uniform([stateSize, actionSize],0,0.01))
      self.outputLayer = tf.matmul(self.inputs1, self.W)
      self.maxOutputNode = tf.argmax(self.outputLayer, 1)
