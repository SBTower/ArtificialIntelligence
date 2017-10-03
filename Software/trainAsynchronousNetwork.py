"""Author: Stuart Tower

Runs an experiment using an asynchronous method. Here there is one global network, and multiple agents each running
on separate threads
"""

from Controller import OrderedController, BatchController
from Environments.AvoidBarriersEnvironment import AvoidBarriersEnvironment
from Environments.MazeEnvironment import MazeEnvironment
from Environments.GymEnvironment import GymEnvironment
from Agents.Agent import Agent
from Agents.Policies.Policy import Policy
from Agents.Policies.Explorers.NoExplorer import NoExplorer
from Agents.Policies.Explorers.EpsilonGreedyExplorer import EpsilonGreedyExplorer
from Agents.Policies.Learners.AsynchronousAdvantageActorCritic import AsynchronousAdvantageActorCritic
from Agents.Policies.Learners.DeepQNetwork import DeepQNetwork
from Networks.FullyConnectedNetwork import FullyConnectedActorCriticNetwork, FullyConnectedDuelingNetwork, FullyConnectedCriticNetwork, FullyConnectedNetwork

import numpy as np
import tensorflow as tf
import copy
import threading

# Define the parameters that describe the expeirment
total_number_of_episodes = 10000
step = 20
rewards = np.zeros(total_number_of_episodes)
avgRewards = []
network_to_load = None
previous_number_of_episodes = 0
save_network_file_name = None
render_environment = True
episode_number = previous_number_of_episodes
epsilon = 1.0
epsilon_decay = 0.0001
epsilon_min = 0.05
batch_size = 30
update_target_rate = None
learning_rate = 0.001
number_of_planning_steps = 0
discount_factor = 0.9
numAgents = 4

# Load an environment, one for each agent to run asynchronously
envs = [GymEnvironment() for i in range(numAgents)]

for env in envs:
    env.num_episodes = previous_number_of_episodes

controllers = []
# Get the possible actions for each environment to run
possibleActions = [env.get_possible_actions() for env in envs]

# Build the global network to feed into each agent
globalNetwork = FullyConnectedActorCriticNetwork(scope='global', state_size=envs[0].get_state_size(),
                                                 action_size=envs[0].get_action_size())

# Define a function that performs the operation to run in each thread, keeping them coordinated
def work(controller, sess, coord):
    with sess.as_default(), sess.graph.as_default():
        while not coord.should_stop():
            for n in range(step):
                controller.run_one_episode()
            coord.request_stop()


with tf.Session() as sess:
    # Load a network if specified
    if network_to_load is not None:
        globalNetwork.load_network(sess, network_to_load)

    for i in range(numAgents):
        # Build a policy, agent and controller for each thread
        this_scope = 'Worker_' + str(i)
        network = FullyConnectedActorCriticNetwork(scope=this_scope, state_size=envs[i].get_state_size(),
                                                 action_size=envs[i].get_action_size())
        # Define the learner to use (must be an asynchronous method)
        learner = AsynchronousAdvantageActorCritic(sess, network=network, learning_rate=learning_rate, discount_factor=discount_factor)
        explorer = EpsilonGreedyExplorer(possible_actions=possibleActions[i], continuous=envs[i].is_action_continuous(),
                                         epsilon=epsilon, epsilon_decay=epsilon_decay, epsilon_min=epsilon_min)
        policy = Policy(learner=copy.copy(learner), explorer=copy.copy(explorer))
        agent = Agent(copy.copy(policy))
        controller = OrderedController(environment=envs[i], agent=copy.copy(agent), batch_size=batch_size,
                                       update_target_rate=update_target_rate)
        controllers.append(copy.copy(controller))

    while episode_number < total_number_of_episodes:

        coord = tf.train.Coordinator()
        # Create a thread for each controller and run them
        worker_threads = []
        for controller in controllers:
            worker_work = lambda: work(controller, sess, coord)
            t = threading.Thread(target=(worker_work))
            t.start()
            worker_threads.append(t)
        coord.join(worker_threads)

        episode_number += step

        if save_network_file_name is not None:
            filename = save_network_file_name + str(episode_number)
            globalNetwork.save_network(sess, filename)

        if render_environment is True:
            controller = controllers[0]
            controller.env.render = True
            r = controller.run_one_episode()
            print(str(episode_number) + ': ' + str(r))
            controller.env.render = False
