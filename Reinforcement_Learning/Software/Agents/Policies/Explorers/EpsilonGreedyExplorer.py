import random


class EpsilonGreedyExplorer:
    def __init__(self, possible_actions, continuous=False, epsilon=0.1, epsilon_decay=0, epsilon_min=0.1):
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.possible_actions = possible_actions
        self.continuous = continuous

    def explore(self):
        if self.epsilon < self.epsilon_min:
            self.epsilon = self.epsilon_min
        if random.random() < self.epsilon:
            return True
        return False

    def get_exploratory_action(self, original_action):
        self.epsilon *= (1 - self.epsilon_decay)
        if self.continuous is False:
            return random.choice(self.possible_actions)
        else:
            actions = [random.uniform(self.possible_actions[0][i], self.possible_actions[1][i]) for i in
                       range(len(self.possible_actions[0]))]
            return actions
