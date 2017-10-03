"""Author: Stuart Tower
"""

from scipy import *
import math
import numpy as np
import random
from .Environment import Environment
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class PMEnv(Environment):
    """An environment representing a predictive maintenance problem
    There are a number of assets that degrade in health, and the timing of maintenance needs to be found by the agent
    """
    def __init__(self, number_of_assets=25, max_episode_length=1000, render=False):
        """Initialise the environment

        :param number_of_assets: The number of assets in the system
        :param max_episode_length: The maximum length of the episode
        :param render: A flag determining whether to render the environment to the screen
        """
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
        """Initialise the health of the assets

        :return: The initial health of all of the assets
        """
        asset_healths = np.zeros(self.numberOfAssets)
        for i in range(len(asset_healths)):
            asset_healths[i] = random.uniform(0.9, 1.0)
        return asset_healths

    def get_state(self):
        """Get the current state of the environment. This is a list of the current health of the assets.

        :return: The current health of all of the assets
        """
        return self.asset_healths

    def get_action_size(self):
        """Get the number of actions available to the agent

        :return: The number of actions available to the agent
        """
        return self.numberOfAssets + 1

    def update(self, action):
        """Update the environment by having the agent perform the input action.
        The agent can either perform maintenance on an asset, or do nothing. Doing maintenance has a cost, and an asset
        with low health also has a cost

        :param action: The action chosen by the agent
        :return: None
        """
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
        """Get the set of possible actions

        :return: The set of possible actions
        """
        possible_actions = range(self.numberOfAssets + 1)
        return possible_actions

    def get_reward(self):
        """Calculate the reward to give to the agent

        :return: The reward to attribute to the agent
        """
        reward = 0
        for health in self.asset_healths:
            if health < 0.75:
                reward = -1 * math.exp(3 * (0.75 - health))
        reward -= self.maintenanceCost
        return reward

    def check_terminal(self):
        """Check if the environment is in a terminal state

        :return: A flag, true if the environment is in a terminal state
        """
        if self.current_episode_length >= self.maxEpisodeLength:
            return True
        else:
            return False

    def reset(self):
        """Reset the environment to the initial state

        :return: None
        """
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
        """Initialise the display for drawing the environment to the screen

        :return: None
        """
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
        """Update the visualisation of the environment on the screen

        :return: None
        """
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
        """Save the current visualisation to an image

        :return: None
        """
        dir_path = os.path.dirname(os.path.realpath(__file__))
        dir_path += '/Videos/PredictiveMaintenanceEnv/Episode-' + str(self.num_episodes)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        self.fig.savefig(dir_path + '/Image_' + str(self.num_figures_saved).zfill(5))
        self.num_figures_saved += 1