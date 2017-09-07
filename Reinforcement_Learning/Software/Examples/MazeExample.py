from Controller import BatchController, OrderedController
from Environments.AvoidBarriersEnvironment import AvoidBarriersEnvironment
from Environments.NavigateToGoalEnvironment import NavigateToGoalEnvironment
from Environments.MazeEnvironment import MazeEnvironment
from Environments.GymEnvironment import GymEnvironment
from Environments.PMEnv import PMEnv
from Agents.Agent import Agent
from Agents.Policies.Policy import Policy
from Agents.Policies.Explorers.EpsilonGreedyExplorer import EpsilonGreedyExplorer
from Agents.Policies.Explorers.NoisyActionExplorer import NoisyActionExplorer
from Agents.Policies.Learners.DeepQNetwork import DeepQNetwork
from Agents.Policies.Learners.QLearning import QLearning
from Agents.Policies.Learners.DDPG import DDPG
from Networks.FullyConnectedNetwork import FullyConnectedDuelingNetwork
from Networks.TabularNetwork import TabularNetwork

import numpy as np
import tensorflow as tf
from gym import wrappers

numGames = 100
step = 5
rewards = np.zeros(numGames)
avgRewards = []
network_to_load = None
previous_number_of_episodes = 0
save_network_file_name = None
render_environment = False
epsilon = 0.05
epsilon_decay = 0.1
epsilon_min = 0.001
batch_size = 1
update_target_rate = None
learning_rate = 0.3  # alpha
number_of_planning_steps = 500
discount_factor = 0.99

env = MazeEnvironment()

env.num_episodes = previous_number_of_episodes

possible_actions = env.get_possible_actions()

network = TabularNetwork(scope='global', state_size=env.get_state_size(), action_size=env.get_action_size())

with tf.Session() as sess:
    if network_to_load is not None:
        network.load_network(sess, network_to_load)

    learner = QLearning(sess, network=network, learning_rate=learning_rate, discount_factor=discount_factor)

    explorer = EpsilonGreedyExplorer(possible_actions=possible_actions, continuous=env.is_action_continuous(),
                                     epsilon=epsilon,
                                     epsilon_decay=epsilon_decay, epsilon_min=epsilon_min)
    policy = Policy(learner=learner, explorer=explorer)
    agent = Agent(policy)
    controller = OrderedController(environment=env, agent=agent, batch_size=batch_size, update_target_rate=update_target_rate)

    for n in range(numGames):
        if n % step == 0 and n >= 0 and render_environment is True:
            controller.env.render = True
        else:
            controller.env.render = False
        r = controller.run_one_episode()
        rewards[n] = r
        print('Episode: ', n, ', Reward: ', r)
        if n % step == 0 and n > 0:
            if save_network_file_name is not None:
                filename = save_network_file_name + str(n + previous_number_of_episodes)
                controller.agent.policy.learner.network.save_network(sess, filename)
            print('Episode: ', n, ', Reward: ', np.mean(rewards[n - step + 1:n + 1]))
            avgRewards.append(np.mean(rewards[n - step + 1:n + 1]))
print(avgRewards)
