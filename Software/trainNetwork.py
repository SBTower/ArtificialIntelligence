"""Author: Stuart Tower

Runs an experiment with the specified environment and agent
Once a successful experiment has been run, the code used should be saved to the 'examples' folder
"""

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
from Networks.FullyConnectedNetwork import FullyConnectedDuelingNetwork, FullyConnectedNetwork, FullyConnectedCriticNetwork, FullyConnectedActorCriticNetwork
from Networks.TabularNetwork import TabularNetwork

import numpy as np
import tensorflow as tf

# Define the parameters that describe the experiment
numGames = 5000
step = 50
rewards = np.zeros(numGames)
avgRewards = []
network_to_load = None
previous_number_of_episodes = 0
save_network_file_name = 'Pendulum'
render_environment = True
epsilon = 1.0
epsilon_decay = 0.00005
epsilon_min = 0.05
batch_size = 200
update_target_rate = 10
learning_rate = 0.0001  # alpha
number_of_planning_steps = 0
discount_factor = 0.99

# Specify the environment to use
env = GymEnvironment('Pendulum-v0')

env.num_episodes = previous_number_of_episodes

possible_actions = env.get_possible_actions()

# Define the networks to use in the policy
actor_network = FullyConnectedNetwork(scope='actor', state_size=env.get_state_size(), action_size=env.get_action_size(), layer_size=[400, 300])
critic_network = FullyConnectedCriticNetwork(scope='critic', state_size=env.get_state_size(), action_size=env.get_action_size())

with tf.Session() as sess:
    # Specify the learner to use
    learner = DDPG(sess, actor_network=actor_network, critic_network=critic_network, learning_rate=learning_rate, discount_factor=discount_factor)
    # Specify the learner to use
    explorer = EpsilonGreedyExplorer(possible_actions=possible_actions, continuous=env.is_action_continuous(),
                                     epsilon=epsilon,
                                     epsilon_decay=epsilon_decay, epsilon_min=epsilon_min)
    # Build the agent, policy and controller
    policy = Policy(learner=learner, explorer=explorer)
    agent = Agent(policy)
    controller = BatchController(environment=env, agent=agent, batch_size=batch_size, update_target_rate=update_target_rate)

    if network_to_load is not None:
        controller.agent.policy.learner.actor_network.load_network(sess, network_to_load)

    # Run the experiment
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
                controller.agent.policy.learner.actor_network.save_network(sess, filename)
            print('Episode: ', n, ', Reward: ', np.mean(rewards[n - step + 1:n + 1]))
            avgRewards.append(np.mean(rewards[n - step + 1:n + 1]))
print(avgRewards)
