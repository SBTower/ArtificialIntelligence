from Controller import OrderedController, BatchController
from Environments import GymEnvironment, MazeEnvironment, GridworldEnvironment, AvoidBarriersEnvironment
from Agents import Agent
from Agents.Policies import Policy
from Agents.Policies.Explorers import EpsilonGreedyExplorer, NoExplorer
from Agents.Policies.Learners import QLearning, DeepQNetwork, AsynchronousAdvantageActorCritic, DDPG
from Networks.FullyConnectedNetwork import FullyConnectedActorCriticNetwork, FullyConnectedDuelingNetwork, FullyConnectedCriticNetwork, FullyConnectedNetwork
import numpy as np
import tensorflow as tf
import copy
import threading
import multiprocessing

env = AvoidBarriersEnvironment(maxEpisodeLength = 100)

with tf.Session() as sess:

  learner = AsynchronousAdvantageActorCritic(sess, scope ='Worker_1', stateSize = env.getStateSize(), actionSize = env.getActionSize(), async = False)
  
  explorer = NoExplorer(possibleActions = env.getPossibleActions(), continuous = env.isActionContinuous())
  policy = Policy(learner = learner, explorer = explorer)
  agent = Agent(policy)
  controller = OrderedController(environment = env, agent = agent, batchSize = 30, updateTargetRate = None)

  controller.render = True
  controller.run_one_episode()
  learner.loadNetwork('AvoidBarriersEnvWorker_1-Network-1200')
