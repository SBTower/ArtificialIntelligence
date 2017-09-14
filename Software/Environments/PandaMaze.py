import copy
import os
import numpy as np
from .Environment import Environment

import matplotlib.pyplot as plt
import matplotlib.patches as patches


class MazeEnvironment(Environment):
    def __init__(self, name='Maze1', render=False):
        self.name = name
        self.maze = self.create_maze(self.name)
        self.start = [1, 1]
        self.goal = [9, 11]
        self.state = copy.copy(self.start)
        self.render = render
        self.num_figures_saved = 0
        self.num_episodes = 0
        self.fig = plt.figure()
        self.initialise_display()
        if self.render is True:
            self.save_figure()
            self.animate()

    def get_state(self):
        return self.enumerate_state(self.state)

    def create_maze(self, name):
        maze = []
        if name is "Maze1":
            maze = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                             [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1],
                             [1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1],
                             [1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1],
                             [1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                             [1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1],
                             [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1],
                             [1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1],
                             [1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
                             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        elif name is "Maze2":
            maze = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                             [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
                             [1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1],
                             [1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                             [1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1],
                             [1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1],
                             [1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1],
                             [1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1],
                             [1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1],
                             [1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1],
                             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        else:
            print
            "Unknown Maze. Loading Default"
            self.create_maze("Maze1")
        return maze

    def load_maze(self, name):
        self.__init__(name)

    def get_reward(self):
        if self.check_terminal():
            reward = 0
        else:
            reward = -1
        return reward

    def update(self, action):
        previous_position = copy.copy(self.state)
        if action == 0:
            self.state[0] -= 1
        elif action == 1:
            self.state[1] -= 1
        elif action == 2:
            self.state[1] += 1
        elif action == 3:
            self.state[0] += 1
        if self.maze[self.state[0]][self.state[1]] == 1:
            self.state = previous_position
        self.animate()

    def enumerate_state(self, state):
        n = len(self.maze) * len(self.maze[0])
        return np.identity(n)[state[0] * len(self.maze) + state[1]]

    def check_terminal(self):
        if self.state == self.goal:
            return True
        return False

    def reset(self):
        self.state = copy.copy(self.start)
        if self.render:
            self.animate()
            self.animate()
            self.animate()
            self.animate()
            self.animate()
            self.animate()
            self.animate()
            self.animate()
            self.animate()
            self.animate()
            self.animate()
            self.animate()
        self.num_episodes += 1
        self.num_figures_saved = 0

    def get_possible_actions(self):
        return [0, 1, 2, 3]

    def initialise_display(self):
        self.ax = self.fig.add_subplot(111)
        for row in range(len(self.maze)):
            for column in range(len(self.maze[0])):
                if [row, column] == self.state:
                    self.position_patch = patches.Rectangle((column, row), 1, 1, facecolor='red')
                    self.ax.add_patch(self.position_patch)
                elif [row, column] == self.goal:
                    self.ax.add_patch(patches.Rectangle((column, row), 1, 1, facecolor="green"))
                elif self.maze[row][column] == 1:
                    self.ax.add_patch(patches.Rectangle((column, row), 1, 1, facecolor="black"))
        self.ax.autoscale()
        self.ax.figure.canvas.draw()

    def animate(self):
        if self.render:
            self.position_patch.set_x(self.state[1])
            self.position_patch.set_y(self.state[0])
            self.ax.figure.canvas.draw()
            # self.fig.show()
            self.save_figure()

    def save_figure(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        dir_path += '/Videos/MazeEnv/Episode-' + str(self.num_episodes)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        self.fig.savefig(dir_path + '/Image_' + str(self.num_figures_saved).zfill(5))
        self.num_figures_saved += 1
