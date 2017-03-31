
from Controller import OrderedController, BatchController
from Environments import GymEnvironment, MazeEnvironment, GridworldEnvironment, AVEnvironment, DoomEnvironment
from Agents import Agent
from Agents.Policies import Policy
from Agents.Policies.Explorers import EpsilonGreedyExplorer, NoExplorer
from Agents.Policies.Learners import QLearning, DeepQNetwork, ConvNetValuePredictor, AdvantageActorCritic, DDPG
from Agents.Policies.Learners.Networks import FullyConnectedActorCriticNetwork, FullyConnectedDuelingNetwork, FullyConnectedCriticNetwork, FullyConnectedNetwork, ConvolutionalActorCriticNetwork
import numpy as np
import tensorflow as tf
import copy
import threading
import multiprocessing

numGames = 100000
step = 100
rewards = np.zeros(numGames)
avgRewards = []

numAgents = 4

envs = [AVEnvironment() for i in xrange(numAgents)]
controllers = []
possibleActions = [env.getPossibleActions() for env in envs]

globalNetwork = FullyConnectedActorCriticNetwork('global',stateSize=envs[0].getStateSize(),actionSize=envs[0].getActionSize())

def work(controller, sess, coord):
  with sess.as_default(), sess.graph.as_default():
    while not coord.should_stop():
      for n in range(100000):
        r = controller.runOneEpisode()
        print controller.agent.policy.learner.scope + ': ' + str(r)
        if n%step == 0 and n > 0:
          filename = 'AVEnv' + controller.agent.policy.learner.scope + '-Network-' + str(n)
          controller.agent.policy.learner.saveNetwork(filename)
      #coord.request_stop()

with tf.Session() as sess:

  for i in range(numAgents):

    this_scope = 'Worker_' + str(i)

    learner = AdvantageActorCritic(sess, scope = copy.copy(this_scope), stateSize = envs[i].getStateSize(), actionSize = envs[i].getActionSize(), async = True)
    explorer = NoExplorer(possibleActions = possibleActions[i], continuous = envs[i].isActionContinuous())
    policy = Policy(learner = copy.copy(learner), explorer = copy.copy(explorer))
    agent = Agent(copy.copy(policy))
    controller = OrderedController(environment = envs[i], agent = copy.copy(agent), batchSize = 30, updateTargetRate = None)
    controllers.append(copy.copy(controller))

  coord = tf.train.Coordinator()

  worker_threads = []
  for controller in controllers:
    worker_work = lambda: work(controller, sess, coord)
    t = threading.Thread(target=(worker_work))
    t.start()
    worker_threads.append(t)
  coord.join(worker_threads)

