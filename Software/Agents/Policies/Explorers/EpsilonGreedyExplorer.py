"""Author: Stuart Tower
"""

import random


class EpsilonGreedyExplorer:
    """An implementation of epsilon greedy exploration.

    Epsilon greedy exploration selects an explorative action with probability epsilon at each step. The value of epsilon
    decays at a rate 'epsilon_decay', until it hits the minimum value of 'epsilon_min'

    """
    def __init__(self, possible_actions, continuous=False, epsilon=0.1, epsilon_decay=0, epsilon_min=0.1):
        """Initialises the parameters for the explorer"""
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.possible_actions = possible_actions
        self.continuous = continuous

    def explore(self):
        """Determines whether or not to explore in the current time step by comparing epsilon to a randomly generated
        number between 0 and 1
        """
        if self.epsilon < self.epsilon_min:
            self.epsilon = self.epsilon_min
        if random.random() < self.epsilon:
            return True
        return False

    def get_exploratory_action(self, original_action):
        """Generates the exploratory action by randomly selecting an action from the set of possible actions"""
        self.epsilon *= (1 - self.epsilon_decay)
        if self.continuous is False:
            return random.choice(self.possible_actions)
        else:
            actions = [random.uniform(self.possible_actions[0][i], self.possible_actions[1][i]) for i in
                       range(len(self.possible_actions[0]))]
            return actions
