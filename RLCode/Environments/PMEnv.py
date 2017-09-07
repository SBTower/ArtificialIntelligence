from scipy import *
import math
import numpy as np
import random
from .Environment import Environment
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class PMEnv(Environment):
    def __init__(self, number_of_assets=25, max_episode_length=1000, render=False):
        self.numberOfAssets = number_of_assets
        self.asset_healths = self.initialise_asset_healths()
        self.assetHealthDecreaseRate = np.zeros(self.numberOfAssets)
        self.maintenanceCost = 0.0
        self.current_episode_length = 0
        self.maxEpisodeLength = max_episode_length
        self.render = render
        self.num_figures_saved = 0
        self.num_episodes = 0
        self.latestAction = None
        self.fig = plt.figure()
        self.initialise_display()
        if self.render is True:
            self.save_figure()
            self.animate()

    def initialise_asset_healths(self):
        asset_healths = np.zeros(self.numberOfAssets)
        for i in range(len(asset_healths)):
            asset_healths[i] = random.uniform(0.9, 1.0)
        return asset_healths

    def get_state(self):
        return self.asset_healths

    def get_action_size(self):
        return self.numberOfAssets + 1

    def update(self, action):
        self.latestAction = action
        self.current_episode_length += 1
        self.maintenanceCost = 0.0
        for i in range(len(self.asset_healths)):
            if i == action:
                self.asset_healths[i] = min(1.0, self.asset_healths[i] + random.uniform(0.5, 1.0))
                self.assetHealthDecreaseRate[i] = 0.0
                self.maintenanceCost = 3.0
            else:
                if self.assetHealthDecreaseRate[i] == 0.0:
                    if random.uniform(0, 1) < 0.005:
                        self.assetHealthDecreaseRate[i] = random.uniform(0.001, 0.2)
                else:
                    self.assetHealthDecreaseRate[i] += random.uniform(-0.03, 0.1)
                self.asset_healths[i] = max(0.0, self.asset_healths[i] - self.assetHealthDecreaseRate[i])
        self.animate()

    def get_possible_actions(self):
        possible_actions = range(self.numberOfAssets + 1)
        return possible_actions

    def get_reward(self):
        reward = 0
        for health in self.asset_healths:
            if health < 0.75:
                reward = -1 * math.exp(3 * (0.75 - health))
        reward -= self.maintenanceCost
        return reward

    def check_terminal(self):
        if self.current_episode_length >= self.maxEpisodeLength:
            return True
        else:
            return False

    def reset(self):
        self.asset_healths = self.initialise_asset_healths()
        self.assetHealthDecreaseRate = np.zeros(self.numberOfAssets)
        self.maintenanceCost = 0.0
        self.current_episode_length = 0
        self.latestAction = None
        if self.render:
            self.animate()
            self.animate()
            self.animate()
            self.animate()
            self.animate()
            self.animate()
        self.num_episodes += 1
        self.num_figures_saved = 0

    def initialise_display(self):
        self.ax = self.fig.add_subplot(111)
        self.current_health_patches = []
        for i in range(len(self.asset_healths)):
            self.current_health_patches.append(patches.Rectangle((i,0),1,self.asset_healths[i],facecolor='blue'))
            self.ax.add_patch(self.current_health_patches[i])
        self.action_patch = patches.Rectangle((-10,-0.2),1,0.2,facecolor='red')
        self.ax.add_patch(self.action_patch)
        self.ax.set_ylim([-0.2, 1])
        self.ax.set_xlim([0, self.numberOfAssets + 1])
        self.ax.figure.canvas.draw()

    def animate(self):
        if self.render:
            for i in range(len(self.current_health_patches)):
                self.current_health_patches[i].set_height(self.asset_healths[i])
            if self.latestAction < self.numberOfAssets:
                self.action_patch.set_x(self.latestAction)
            else:
                self.action_patch.set_x(-10)
            self.ax.figure.canvas.draw()
            self.fig.show()
            self.save_figure()

    def save_figure(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        dir_path += '/Videos/PredictiveMaintenanceEnv/Episode-' + str(self.num_episodes)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        self.fig.savefig(dir_path + '/Image_' + str(self.num_figures_saved).zfill(5))
        self.num_figures_saved += 1