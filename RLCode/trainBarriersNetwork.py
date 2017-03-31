
#from Controller import OrderedController, BatchController
#from Environments import GymEnvironment, MazeEnvironment, GridworldEnvironment, AVEnvironment, #PredictiveMaintenanceEnvironment, PMEnv, AvoidBarriersEnvironment, NavigateToGoalEnvironment
#from Agents import Agent
#from Agents.Policies import Policy
#from Agents.Policies.Explorers import EpsilonGreedyExplorer, NoisyActionExplorer
#from Agents.Policies.Learners import QLearning, DeepQNetwork, ConvNetValuePredictor, DDPG
#import numpy as np
#import tensorflow as tf

#numGames = 100000
#step = 100
#rewards = np.zeros(numGames)
#avgRewards = []

#env = GymEnvironment('CartPole-v0')
#env = AvoidBarriersEnvironment()
#env = NavigateToGoalEnvironment(maxEpisodeLength = 100)
#env = MazeEnvironment('Maze1')
#env = AVEnvironment()
#possibleActions = env.getPossibleActions()

#with tf.Session() as sess:

#  learner = DeepQNetwork(sess, scope = 'global', stateSize = env.getStateSize(), actionSize = env.getActionSize(), async = False)
#  explorer = EpsilonGreedyExplorer(possibleActions = possibleActions, continuous = env.isActionContinuous(), epsilon = 1, epsilonDecay = 0.00002, epsilonMin = 0.05)
#  policy = Policy(learner = learner, explorer = explorer)
#  agent = Agent(policy)
#  controller = BatchController(environment = env, agent = agent, batchSize = 500, updateTargetRate = 1)

#  for n in range(numGames):
#    r = controller.runOneEpisode()
#    controller.render = False
#    rewards[n] = r
#    print 'Episode: ', n, ', Reward: ', r
#    if n%step == 0 and n > 0:
#      controller.render = True
#      #filename = 'AutoVehicleTwo-' + str(n)
#      #controller.agent.policy.learner.saveNetwork(filename)
#      print 'Episode: ', n, ', Reward: ', np.mean(rewards[n-step:n])
#      print controller.agent.policy.explorer.epsilon
#      avgRewards.append(np.mean(rewards[n-step:n]))
#  print avgRewards


from Controller import OrderedController, BatchController
from Environments import GymEnvironment, MazeEnvironment, GridworldEnvironment, AVEnvironment, DoomEnvironment, AvoidBarriersEnvironment
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

envs = [AvoidBarriersEnvironment(maxEpisodeLength = 100) for i in xrange(numAgents)]
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
          filename = 'AvoidBarriersEnv' + controller.agent.policy.learner.scope + '-Network-' + str(n)
          controller.agent.policy.learner.saveNetwork(filename)
      #coord.request_stop()

with tf.Session() as sess:

  for i in range(numAgents):

    this_scope = 'Worker_' + str(i)

    learner = AdvantageActorCritic(sess, scope = copy.copy(this_scope), stateSize = envs[i].getStateSize(), actionSize = envs[i].getActionSize(), async = True)
    #learner.loadNetwork('AvoidBarriersEnv' + learner.scope + '-Network-' + str(1200))
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

