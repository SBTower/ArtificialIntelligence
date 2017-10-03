"""Author: Stuart Tower
"""

from scipy import *
import copy
import math
import numpy as np
import random
import shapely
import shapely.geometry
import os
from .Vehicles.SimpleVehicle import SimpleVehicle
from .Environment import Environment

import matplotlib.pyplot as plt
import matplotlib.patches as patches


class AvoidBarriersEnvironment(Environment):
    """An environment that models an autonomous vehicle that has to move around an environment without hitting any of
    the obstacles. The environment uses a very simple model of the movement of the boat, assuming the agent can directly
    control acceleration (linear and angular)

    """
    def __init__(self, width=200, height=200, max_episode_length=100, render=False, save_video = False, number_of_obstacles = 4):
        """Initialise the environment

        :param width: The width of the playground the boat can explore
        :param height: The height of the playground the boat can explore
        :param max_episode_length: The maximum length of the episode
        :param render: A flag, if true the environment will be drawn to the screen
        :param save_video: A flag, if true the environment will save a video to a file
        :param number_of_obstacles: The number of obstacles to place in the playground
        """
        self.width = width
        self.height = height
        self.max_episode_length = max_episode_length
        self.num_figures_saved = 0
        self.num_episodes = 0   # A count of the number of episodes completed
        self.render = render
        self.save_video = save_video
        self.current_episode_length = 0
        self.number_of_obstacles = number_of_obstacles

        # Randomly initialise the position and orientation of the boat
        pos_x = self.width / 10 + random.random() * 4 * self.width / 5
        pos_y = self.height / 10 + random.random() * 4 * self.height / 5
        orientation = random.random() * 2 * math.pi
        self.boat = SimpleVehicle([pos_x, pos_y, orientation])
        # Generate the barriers
        self.barriers = self.create_barriers()
        # Randomly re-position the boat until it doesn't overlap with the barriers
        while self.has_crashed() is True:
            pos_x = self.width / 10 + random.random() * 4 * self.width / 5
            pos_y = self.height / 10 + random.random() * 4 * self.height / 5
            orientation = random.random() * 2 * math.pi
            self.boat.reset_position([pos_x, pos_y, orientation])
        self.fig = plt.figure()
        self.initialise_display()
        if self.render is True:
            self.save_figure()
            self.animate()

    def create_barriers(self):
        """Generate the barriers that the boat has to avoid

        :return: The barriers that describe both the edge of the playground and the obstacles within the environment
        """
        # Generate the barriers around the outside of the playground
        barrier1 = [[0, 5], [self.width, 5], [self.width, 0], [0, 0]]
        barrier2 = [[0, self.height - 5], [self.width, self.height - 5], [self.width, self.height], [0, self.height]]
        barrier3 = [[self.width - 5, self.height], [self.width, self.height], [self.width, 0], [self.width - 5, 0]]
        barrier4 = [[5, self.height], [0, self.height], [0, 0], [5, 0]]

        barriers = [barrier1, barrier2, barrier3, barrier4]

        # Generate the barriers describing the obstacles within the playground
        for i in range(self.number_of_obstacles):
            random_start_x = random.uniform(20, self.width - 20)
            random_start_y = random.uniform(20, self.height - 20)
            point1 = [random_start_x, random_start_y]
            point2 = [point1[0] + random.uniform(15, 40), point1[1]]
            point3 = [point2[0], point2[1] + random.uniform(15, 40)]
            point4 = [point3[0] + random.uniform(-40, -15), point3[1]]
            barriers.append([point1, point2, point3, point4])

        return barriers

    def get_state(self):
        """Gets the state of the boat, which includes the readings from a number of range sensors positioned at the
        corners of the boat

        :return: The readings from each of the boats sensors
        """
        # Position the sensors at the corners of the boat
        sensor_locations = np.array(self.boat.outline)
        sensor_readings = []

        # Use trigonometry to calculate the distance from the sensor source to the nearest barrier
        for i in range(len(sensor_locations)):

            if i == 0:
                sign_1 = -1
                sign_2 = 1
                sign_3 = -1
                sign_4 = -1
            elif i == 1:
                sign_1 = -1
                sign_2 = 1
                sign_3 = 1
                sign_4 = 1
            elif i == 2:
                sign_1 = 1
                sign_2 = -1
                sign_3 = 1
                sign_4 = 1
            elif i == 3:
                sign_1 = 1
                sign_2 = -1
                sign_3 = -1
                sign_4 = -1

            sensor_readings.append(self.boat.sensor_range)
            sensor_readings.append(self.boat.sensor_range)
            x1 = sensor_locations[i][0]
            y1 = sensor_locations[i][1]

            for barrier in self.barriers:

                x2 = x1 + sign_1 * self.boat.sensor_range * math.cos(self.boat.pos[2])
                y2 = y1 + sign_2 * self.boat.sensor_range * math.sin(self.boat.pos[2])

                x3 = x1 + sign_3 * self.boat.sensor_range * math.sin(self.boat.pos[2])
                y3 = y1 + sign_4 * self.boat.sensor_range * math.cos(self.boat.pos[2])

                line1 = shapely.geometry.LineString([[x1, y1], [x2, y2]])
                line2 = shapely.geometry.LineString([[x1, y1], [x3, y3]])

                barrier_polygon = shapely.geometry.Polygon(np.array(barrier))
                sd = self.boat.sensor_range
                while line1.intersects(barrier_polygon) and sd > 0:
                    sd -= 1
                    x2 = x1 + sign_1 * sd * math.cos(self.boat.pos[2])
                    y2 = y1 + sign_2 * sd * math.sin(self.boat.pos[2])
                    line1 = shapely.geometry.LineString([[x1, y1], [x2, y2]])
                if sd < sensor_readings[2 * i]:
                    sensor_readings[2 * i] = sd

                sd = self.boat.sensor_range
                while line2.intersects(barrier_polygon) and sd > 0:
                    sd -= 1
                    x3 = x1 + sign_3 * sd * math.sin(self.boat.pos[2])
                    y3 = y1 + sign_4 * sd * math.cos(self.boat.pos[2])
                    line2 = shapely.geometry.LineString([[x1, y1], [x3, y3]])
                if sd < sensor_readings[2 * i + 1]:
                    sensor_readings[2 * i + 1] = sd

        # Include the speed and velocity of the boat in the state of the environment
        sensor_readings.append(copy.copy(self.boat.speed))
        sensor_readings.append(copy.copy(self.boat.angularVelocity))

        return sensor_readings

    def update(self, action):
        """Update the environment based on the action the agent has taken. The action is a discretisation containing the
        linear and angular acceleration

        :param action: The action the agent has taken
        :return: None
        """
        self.current_episode_length += 1
        n1 = action % 11
        n2 = action / 11
        action1 = float((n1 - 5)) / 2   # Linear acceleration
        action2 = float((n2 - 5)) / 200 # Angular acceleration
        timestep = 0.5
        self.boat.change_acceleration(action1)
        self.boat.change_angular_acceleration(action2)
        self.boat.update_position(timestep)
        self.animate()

    def get_possible_actions(self):
        """Returns the list of actions available to the agent

        :return: Set of possible actions
        """
        possible_actions = range(121)
        return possible_actions

    def get_reward(self):
        """Gets the reward to give the agent based on the current state

        :return: The reward to the agent in the current state
        """
        if self.has_crashed():
            reward = -1000
        else:
            reward = 1
        return reward

    def check_terminal(self):
        """Checks if the current state is a terminal state

        :return: A boolean which is true if the current state is terminal
        """
        if self.has_crashed():
            return True
        elif self.current_episode_length> self.max_episode_length:
            return True
        else:
            return False

    def has_crashed(self):
        """Checks if the boat has crashed into a barrier

        :return: A boolean which is true if the boat has crashed
        """
        crashed = False
        boat = np.array(self.boat.outline)
        boat_polygon = shapely.geometry.Polygon(boat)
        for barrier in self.barriers:
            barrier_polygon = shapely.geometry.Polygon(barrier)
            if barrier_polygon.intersects(boat_polygon):
                crashed = True
        return crashed

    def reset(self):
        """Resets the current environment

        :return: None
        """
        self.current_episode_length = 0
        pos_x = self.width / 10 + random.random() * 4 * self.width / 5
        pos_y = self.height / 10 + random.random() * 4 * self.height / 5
        orientation = random.random() * 2 * math.pi
        self.boat.reset_position([pos_x, pos_y, orientation])
        self.barriers = self.create_barriers()
        while self.has_crashed() is True:
            pos_x = self.width / 10 + random.random() * 4 * self.width / 5
            pos_y = self.height / 10 + random.random() * 4 * self.height / 5
            orientation = random.random() * 2 * math.pi
            self.boat.reset_position([pos_x, pos_y, orientation])
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
        """Initialises the display using matplotlib

        :return: None
        """
        self.ax = self.fig.add_subplot(111)
        self.barrier_patches = []
        for i in range(len(self.barriers)):
            self.barrier_patches.append(patches.Polygon(self.barriers[i]))
            self.ax.add_patch(self.barrier_patches[i])
        self.boat_patch = patches.Polygon(self.boat.outline)
        self.ax.add_patch(self.boat_patch)
        self.ax.set_ylim([0, self.height])
        self.ax.set_xlim([0, self.width])
        self.ax.figure.canvas.draw()

    def animate(self):
        """Updates the display to represent the current state and saves an image if specified

        :return: None
        """
        if self.render:
            for i in range(len(self.barriers[4:])):
                self.barrier_patches[i+4].set_xy(self.barriers[i+4])
            self.boat_patch.set_xy(self.boat.outline)
            self.ax.figure.canvas.draw()
            self.fig.show()
            if self.save_video:
                self.save_figure()

    def save_figure(self):
        """Saves the current visualisation as an image

        :return: None
        """
        dir_path = os.path.dirname(os.path.realpath(__file__))
        dir_path += '/Videos/AvoidBarriersEnv/Episode-' + str(self.num_episodes)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        self.fig.savefig(dir_path + '/Image_' + str(self.num_figures_saved).zfill(5))
        self.num_figures_saved += 1

