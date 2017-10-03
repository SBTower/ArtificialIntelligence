"""Author: Stuart Tower
"""

import copy
import os
import numpy as np
from .Environment import Environment

import matplotlib.pyplot as plt
import matplotlib.patches as patches


class MazeEnvironment(Environment):
    """An environment describing a simple maze, where the agent has to navigate from the start square to the end square
    """
    def __init__(self, name='Maze1', render=False):
        """Initialise the maze environment

        :param name: The name of the maze to load. Options are 'Maze1' or 'Maze2'
        :param render: A flag determining whether to render the environment to the screen
        """
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
        """Get the enumerated representation of the current state

        :return: The current state, enumerated for input to the agent policy
        """
        return self.enumerate_state(self.state)

    def create_maze(self, name):
        """Create the staructure of the maze to navigate

        :param name: The name of the maze to create
        :return: The structure of the maze
        """
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
        """Reset the environment with the specified maze

        :param name: Name of the maze to load
        :return: None
        """
        self.__init__(name)

    def get_reward(self):
        """Get the reward to attribute to the agent based on the current state

        :return: The reward to attribute to the agent
        """
        if self.check_terminal():
            reward = 0
        else:
            reward = -1
        return reward

    def update(self, action):
        """Update the state of the environment based on the agent taking the input state

        :param action: The action taken by the agent
        :return: None
        """
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
        """Enumerate the state for input to the agent policy

        :param state: The internal representation of the environment state
        :return: The enumerated representation of the environment for input to the agent policy
        """
        n = len(self.maze) * len(self.maze[0])
        return np.identity(n)[state[0] * len(self.maze) + state[1]]

    def check_terminal(self):
        """Check if the environment is in a terminal state

        :return: A flag, true if the environment is in a terminal state
        """
        if self.state == self.goal:
            return True
        return False

    def reset(self):
        """Draw the enviornment to the screen

        :return: None
        """
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
            self.animate()  # Lots of calls to animate to control speed of rendering
        self.num_episodes += 1
        self.num_figures_saved = 0

    def get_possible_actions(self):
        """Get the set of possible actions an agent can take

        :return: The set of possible actions an agent can take
        """
        return [0, 1, 2, 3]

    def initialise_display(self):
        """Initialise the display used for rendering the environment to the screen

        :return: None
        """
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
        """Update the visualisation of the environment

        :return: None
        """
        if self.render:
            self.position_patch.set_x(self.state[1])
            self.position_patch.set_y(self.state[0])
            self.ax.figure.canvas.draw()
            self.fig.show()
            self.save_figure()

    def save_figure(self):
        """Save the current visualisation of the environment as an image

        :return: None
        """
        dir_path = os.path.dirname(os.path.realpath(__file__))
        dir_path += '/Videos/MazeEnv/Episode-' + str(self.num_episodes)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        self.fig.savefig(dir_path + '/Image_' + str(self.num_figures_saved).zfill(5))
        self.num_figures_saved += 1
