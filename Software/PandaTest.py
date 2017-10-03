"""Author: Stuart Tower

This experiment uses the Panda3D environment to test the integration between the game engine and the reinforcement
learning code
"""

from Controller import BatchController, OrderedController
from Environments.PandaMaze import PandaMaze
from Agents.Agent import Agent
from Agents.Policies.Policy import Policy
from Agents.Policies.Explorers.EpsilonGreedyExplorer import EpsilonGreedyExplorer
from Agents.Policies.Learners.QLearning import QLearning
from Networks.FullyConnectedNetwork import FullyConnectedDuelingNetwork, FullyConnectedNetwork, FullyConnectedCriticNetwork, FullyConnectedActorCriticNetwork
from Networks.TabularNetwork import TabularNetwork

import numpy as np
import tensorflow as tf

# Define the parameters that describe the experiment to run
numGames = 5000
step = 50
rewards = np.zeros(numGames)
avgRewards = []
network_to_load = None
previous_number_of_episodes = 0
save_network_file_name = None
render_environment = False
epsilon = 1.0
epsilon_decay = 0.005
epsilon_min = 0.005
batch_size = 200
update_target_rate = None
learning_rate = 0.0001  # alpha
number_of_planning_steps = 0
discount_factor = 0.99

# Define the environment
env = PandaMaze()
print(env.get_action_size())
env.num_episodes = previous_number_of_episodes

# Get the set of possible actions
possible_actions = env.get_possible_actions()

# Define the network to use in the policy
network = FullyConnectedNetwork(scope='actor', state_size=env.get_state_size(), action_size=env.get_action_size(), layer_size=[400, 300])

with tf.Session() as sess:
    # Define the learner as a 'Q-Learning' policy
    learner = QLearning(sess, network=network, learning_rate=learning_rate, discount_factor=discount_factor)
    # Use an epsilon greedy explorer
    explorer = EpsilonGreedyExplorer(possible_actions=possible_actions, continuous=env.is_action_continuous(),
                                     epsilon=epsilon,
                                     epsilon_decay=epsilon_decay, epsilon_min=epsilon_min)
    policy = Policy(learner=learner, explorer=explorer)
    agent = Agent(policy)
    controller = BatchController(environment=env, agent=agent, batch_size=batch_size, update_target_rate=update_target_rate)

    # Load a previous network if specified
    if network_to_load is not None:
        controller.agent.policy.learner.actor_network.load_network(sess, network_to_load)

    # Run the experiment
    for n in range(numGames):
        # Render every 'step' episodes
        if n % step == 0 and n >= 0 and render_environment is True:
            controller.env.render = True
        else:
            controller.env.render = False
        r = controller.run_one_episode()
        rewards[n] = r
        print('Episode: ', n, ', Reward: ', r)
        # Save the network if triggered
        if n % step == 0 and n > 0:
            if save_network_file_name is not None:
                filename = save_network_file_name + str(n + previous_number_of_episodes)
                controller.agent.policy.learner.actor_network.save_network(sess, filename)
            print('Episode: ', n, ', Reward: ', np.mean(rewards[n - step + 1:n + 1]))
            avgRewards.append(np.mean(rewards[n - step + 1:n + 1]))
print(avgRewards)
