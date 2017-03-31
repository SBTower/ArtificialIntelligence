
from Controller import OrderedController, BatchController
from Environments import GymEnvironment, MazeEnvironment, GridworldEnvironment, AVEnvironment, PredictiveMaintenanceEnvironment, PMEnv, AvoidBarriersEnvironment, NavigateToGoalEnvironment
from Agents import Agent
from Agents.Policies import Policy
from Agents.Policies.Explorers import EpsilonGreedyExplorer, NoisyActionExplorer
from Agents.Policies.Learners import QLearning, DeepQNetwork, ConvNetValuePredictor, DDPG
import numpy as np
import tensorflow as tf

numGames = 10000
step = 10
rewards = np.zeros(numGames)
avgRewards = []

#env = GymEnvironment('CartPole-v0')
#env = AvoidBarriersEnvironment()
env = NavigateToGoalEnvironment(maxEpisodeLength = 100)
#env = MazeEnvironment('Maze1')
#env = AVEnvironment()
possibleActions = env.getPossibleActions()

with tf.Session() as sess:

  learner = DeepQNetwork(sess, scope = 'global', stateSize = env.getStateSize(), actionSize = env.getActionSize(), async = False)
  explorer = EpsilonGreedyExplorer(possibleActions = possibleActions, continuous = env.isActionContinuous(), epsilon = 1, epsilonDecay = 0.0001, epsilonMin = 0.05)
  policy = Policy(learner = learner, explorer = explorer)
  agent = Agent(policy)
  controller = BatchController(environment = env, agent = agent, batchSize = 500, updateTargetRate = 1)

  for n in range(numGames):
    r = controller.runOneEpisode()
    controller.render = False
    rewards[n] = r
    print 'Episode: ', n, ', Reward: ', r
    if n%step == 0 and n > 0:
      controller.render = True
      #filename = 'AutoVehicleTwo-' + str(n)
      #controller.agent.policy.learner.saveNetwork(filename)
      print 'Episode: ', n, ', Reward: ', np.mean(rewards[n-step:n])
      print controller.agent.policy.explorer.epsilon
      avgRewards.append(np.mean(rewards[n-step:n]))
  print avgRewards

