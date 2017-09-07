from scipy import *
import pylab
import copy
import math
import numpy as np
import random
from .Environment import Environment


class SearchAreaEnvironment(Environment):
    def __init__(self, number_of_assets=3, height=100, width=100):
        self.number_of_assets = number_of_assets
        self.height = height
        self.width = width
        self.area = np.zeros(height, width)
        self.assetHealths = []
        self.assetHealthDecreaseRate = []
        for _ in range(self.number_of_assets):
            self.assetHealths.append(0.0)
            self.assetHealthDecreaseRate.append(0.0)
        self.maxEpisodeLength = maxEpisodeLength
        self.done = False
        self.latest_action = None
        self.maintenance_cost = 0.0
        self.fig = plt.figure()

    def get_state(self):
        return self.area

    def get_action_size(self):
        return self.number_of_assets

    def update(self, action):
        self.latest_action = action
        self.count += 1
        self.maintenance_cost = 0.0
        for i in range(len(self.assetHealths)):
            if i == action:
                self.assetHealths[i] = min(1.0, self.assetHealths[i] + random.uniform(0.5, 1.0))
                self.assetHealthDecreaseRate[i] = 0.0
                self.maintenance_cost = 3.0
            else:
                if self.assetHealthDecreaseRate[i] == 0.0:
                    if random.uniform(0, 1) < 0.005:
                        self.assetHealthDecreaseRate[i] = random.uniform(0.001, 0.2)
                else:
                    self.assetHealthDecreaseRate[i] += random.uniform(-0.03, 0.1)
                self.assetHealths[i] = max(0.0, self.assetHealths[i] - self.assetHealthDecreaseRate[i])

    def get_possible_actions(self):
        possible_actions = range(self.number_of_assets + 1)
        return possible_actions

    def get_reward(self):
        reward = 0
        for health in self.assetHealths:
            if health < 0.75:
                reward = -1 * math.exp(5 * (0.75 - health))
        reward -= self.maintenance_cost
        return reward

    def check_terminal(self):
        if self.count >= self.maxEpisodeLength:
            return True
        else:
            return False

    def reset(self):
        self.__init__(number_of_assets=self.number_of_assets, maxEpisodeLength=self.maxEpisodeLength)

    def render(self):
        if self.count % 1 == 0:
            self.fig.clear()
            ax = self.fig.add_subplot(111)
            ax.bar(range(self.number_of_assets), self.assetHealths)
            if self.latest_action is not None:
                if self.latest_action < self.number_of_assets:
                    ax.bar(self.latest_action, -0.2, color='red')
            ax.set_ylim([-0.2, 1])
            ax.set_xlim([0, self.number_of_assets + 1])
            ax.figure.canvas.draw()
            self.fig.show()
        if self.checkTerminal():
            plt.close(self.fig)
            self.fig = plt.figure()
