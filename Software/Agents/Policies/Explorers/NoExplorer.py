"""Author: Stuart Tower
"""

class NoExplorer:
    """An explorer class that never performs an exploratory action"""
    def __init__(self, possible_actions, continuous=False, epsilon=0.1, epsilon_decay=0, epsilon_min=0.1):
        self.epsilon = epsilon
        self.epsilonDecay = epsilon_decay
        self.epsilonMin = epsilon_min
        self.possibleActions = possible_actions
        self.continuous = continuous

    def explore(self):
        return False

    def get_exploratory_action(self, original_action):
        return original_action
