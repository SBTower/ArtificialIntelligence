"""Author: Stuart Tower
"""

import numpy as np


class NoisyActionExplorer:
    """An implementation of a noisy action explorer.

    Designed for continuous action examples, this exploration method adds a certain amount of noise to the original
    action, sampled from a normal distribution. The standard deviation of the distribution is epsilon, and this decays
    at a rate of epsilon_decay until it reaches a value of epsilon_min
    """
    def __init__(self, possible_actions, continuous=False, epsilon=0.1, epsilon_decay=0., epsilon_min=0.):
        """Initialises the parameters for the explorer"""
        self.possible_actions = possible_actions
        self.continuous = continuous
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    def explore(self):
        """Determines whether or not to explore in the current time step. If epsilon is less than epsilon_min then no
        exploration is necessary
        """
        if self.epsilon < self.epsilon_min:
            return False
        else:
            return True

    def get_exploratory_action(self, original_action):
        """Generates the exploratory action. If the action space is discrete, then the chosen action is rounded to the
        nearest integer value. Otherwise the noise is simply added to the continuous action value
        """
        self.epsilon *= (1 - self.epsilon_decay)
        if self.continuous is False:
            action = int(round(original_action + np.random.normal(0, self.epsilon * len(self.possible_actions) / 5)))
            if action < min(self.possible_actions):
                action = min(self.possible_actions)
            if action > max(self.possible_actions):
                action = max(self.possible_actions)
        else:
            action = []
            for action_index in range(len(original_action)):
                new_act = original_action[action_index] + np.random.normal(0, self.epsilon)
                if new_act > self.possible_actions[1][action_index]:
                    new_act = self.possible_actions[1][action_index]
                elif new_act < self.possible_actions[0][action_index]:
                    new_act = self.possible_actions[0][action_index]
                action.append(new_act)
        return action
