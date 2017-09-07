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
    def __init__(self, width=200, height=200, max_episode_length=100, render=False, save_video = False):
        self.width = width
        self.height = height
        self.max_episode_length = max_episode_length
        self.num_figures_saved = 0
        self.num_episodes = 0
        self.render = render
        self.save_video = save_video
        self.current_episode_length = 0
        self.number_of_obstacles = 4
        pos_x = self.width / 10 + random.random() * 4 * self.width / 5
        pos_y = self.height / 10 + random.random() * 4 * self.height / 5
        orientation = random.random() * 2 * math.pi
        self.boat = SimpleVehicle([pos_x, pos_y, orientation])
        self.barriers = self.create_barriers()
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
        barrier1 = [[0, 5], [self.width, 5], [self.width, 0], [0, 0]]
        barrier2 = [[0, self.height - 5], [self.width, self.height - 5], [self.width, self.height], [0, self.height]]
        barrier3 = [[self.width - 5, self.height], [self.width, self.height], [self.width, 0], [self.width - 5, 0]]
        barrier4 = [[5, self.height], [0, self.height], [0, 0], [5, 0]]

        barriers = [barrier1, barrier2, barrier3, barrier4]

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
        sensor_locations = np.array(self.boat.outline)
        sensor_readings = []
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

        sensor_readings.append(copy.copy(self.boat.speed))
        sensor_readings.append(copy.copy(self.boat.angularVelocity))

        return sensor_readings

    def update(self, action):
        self.current_episode_length += 1
        n1 = action % 11
        n2 = action / 11
        action1 = float((n1 - 5)) / 2
        action2 = float((n2 - 5)) / 200
        timestep = 0.5
        self.boat.change_acceleration(action1)
        self.boat.change_angular_acceleration(action2)
        self.boat.update_position(timestep)
        self.animate()

    def get_possible_actions(self):
        possible_actions = range(121)
        return possible_actions

    def get_reward(self):
        if self.has_crashed():
            reward = -1000
        else:
            reward = 1
        return reward

    def check_terminal(self):
        if self.has_crashed():
            return True
        elif self.current_episode_length> self.max_episode_length:
            return True
        else:
            return False

    def has_crashed(self):
        crashed = False
        boat = np.array(self.boat.outline)
        boat_polygon = shapely.geometry.Polygon(boat)
        for barrier in self.barriers:
            barrier_polygon = shapely.geometry.Polygon(barrier)
            if barrier_polygon.intersects(boat_polygon):
                crashed = True
        return crashed

    def reset(self):
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
        if self.render:
            for i in range(len(self.barriers[4:])):
                self.barrier_patches[i+4].set_xy(self.barriers[i+4])
            self.boat_patch.set_xy(self.boat.outline)
            self.ax.figure.canvas.draw()
            self.fig.show()
            if self.save_video:
                self.save_figure()

    def save_figure(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        dir_path += '/Videos/AvoidBarriersEnv/Episode-' + str(self.num_episodes)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        self.fig.savefig(dir_path + '/Image_' + str(self.num_figures_saved).zfill(5))
        self.num_figures_saved += 1

