import random
import numpy as np


class NoisyActionExplorer:
    def __init__(self, possible_actions, continuous=False, epsilon=0.1, epsilon_decay=0., epsilon_min=0.):
        self.possible_actions = possible_actions
        self.continuous = continuous
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    def explore(self):
        if self.epsilon < self.epsilon_min:
            return False
        else:
            return True

    def get_exploratory_action(self, original_action):
        self.epsilon *= (1 - self.epsilon_decay)
        if self.continuous is False:
            action = int(round(original_action + np.random.normal(0, self.epsilon * len(self.possible_actions) / 5)))
            if action < min(self.possible_actions):
                action = min(self.possible_actions)
            if action > max(self.possible_actions):
                action = max(self.possible_actions)
        else:
            action = []
            for i in range(len(original_action)):
                new_act = original_action[i] + np.random.normal(0, self.epsilon)
                if new_act > self.possible_actions[1][i]:
                    new_act = self.possible_actions[1][i]
                elif new_act < self.possible_actions[0][i]:
                    new_act = self.possible_actions[0][i]
                action.append(new_act)
        return action
