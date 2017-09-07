from scipy import *
import copy
import math
import numpy as np
import random
from .Vehicles.SimpleVehicle import SimpleVehicle
from .Environment import Environment
from shapely.geometry import Point
from shapely.geometry import LineString

import matplotlib.pyplot as plt
import matplotlib.patches as patches


class AVEnvironment(Environment):
    def __init__(self, width=200, height=200):
        self.width = width
        self.height = height
        pos_x = self.width / 10 + random.random() * 4 * self.width / 5
        pos_y = self.height / 10 + random.random() * 4 * self.height / 5
        orientation = random.random() * 2 * math.pi
        self.boat = SimpleVehicle([pos_x, pos_y, orientation])
        self.barriers = self.create_barriers()
        self.create_goal()
        self.reward = 0.0
        self.done = False
        # self.fig = plt.figure()
        self.count = 0
        self.timestep = 0
        self.maxTimeStep = 100

    def create_barriers(self):
        barrier1 = [[0, 5], [self.width, 5], [self.width, 0], [0, 0]]
        barrier2 = [[0, self.height - 5], [self.width, self.height - 5], [self.width, self.height], [0, self.height]]
        barrier3 = [[self.width - 5, self.height], [self.width, self.height], [self.width, 0], [self.width - 5, 0]]
        barrier4 = [[5, self.height], [0, self.height], [0, 0], [5, 0]]

        barriers = [barrier1, barrier2, barrier3, barrier4]
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

                barrier_polygons = shapely.geometry.Polygon(np.array(barrier))
                sd = self.boat.sensor_range
                while line1.intersects(barrier_polygons) and sd > 0:
                    sd -= 1
                    x2 = x1 + sign_1 * sd * math.cos(self.boat.pos[2])
                    y2 = y1 + sign_2 * sd * math.sin(self.boat.pos[2])
                    line1 = shapely.geometry.LineString([[x1, y1], [x2, y2]])
                if sd < sensor_readings[2 * i]:
                    sensor_readings[2 * i] = sd

                sd = self.boat.sensor_range
                while line2.intersects(barrier_polygons) and sd > 0:
                    sd -= 1
                    x3 = x1 + sign_3 * sd * math.sin(self.boat.pos[2])
                    y3 = y1 + sign_4 * sd * math.cos(self.boat.pos[2])
                    line2 = shapely.geometry.LineString([[x1, y1], [x3, y3]])
                if sd < sensor_readings[2 * i + 1]:
                    sensor_readings[2 * i + 1] = sd

        # sensor_readings.append(copy.copy(self.boat.pos[0]))
        # sensor_readings.append(copy.copy(self.boat.pos[1]))
        sensor_readings.append(copy.copy(self.boat.speed))
        sensor_readings.append(copy.copy(self.boat.angularVelocity))
        sensor_readings.append(self.get_distance_to_goal())
        sensor_readings.append(self.get_angle_to_goal())

        return sensor_readings

    def create_goal(self):
        goal_x = self.width / 10 + random.random() * 4 * self.width / 5
        goal_y = self.height / 10 + random.random() * 4 * self.height / 5
        self.goal = [goal_x, goal_y]

    def get_distance_to_goal(self):
        dist_x = self.goal[0] - self.boat.pos[0]
        dist_y = self.goal[1] - self.boat.pos[1]
        distance = math.sqrt(dist_x * dist_x + dist_y * dist_y)
        return distance

    def get_angle_to_goal(self):
        dist_x = self.goal[0] - self.boat.pos[0]
        dist_y = self.goal[1] - self.boat.pos[1]

        if dist_y == 0:
            if dist_x == 0:
                angle_to_goal = 0
            elif dist_x > 0:
                angle_to_goal = math.pi / 2
            else:
                angle_to_goal = 3 * math.pi / 2
        elif dist_y > 0:
            if dist_x >= 0:
                angle_to_goal = math.atan(abs(dist_x) / abs(dist_y))
            else:
                angle_to_goal = 2 * math.pi - math.atan(abs(dist_x) / abs(dist_y))
        else:
            if dist_x >= 0:
                angle_to_goal = math.pi - math.atan(abs(dist_x) / abs(dist_y))
            else:
                angle_to_goal = math.pi + math.atan(abs(dist_x) / abs(dist_y))
        return angle_to_goal

    def update(self, action):
        self.timestep += 1
        n1 = action % 11
        n2 = action / 11
        action1 = float((n1 - 5)) / 2
        action2 = float((n2 - 5)) / 200
        timestep = 1
        self.boat.change_acceleration(action1)
        self.boat.change_angular_acceleration(action2)
        self.boat.update_position(timestep)
        if self.check_terminal():
            self.done = True
        else:
            self.done = False

    def get_possible_actions(self):
        possible_actions = range(121)
        return possible_actions

    def get_reward(self):
        if self.has_crashed():
            reward = -1000
        elif self.check_in_goal():
            reward = 1000
        else:
            reward = -1
        return reward

    def check_terminal(self):
        if self.has_crashed():
            return True
        elif self.check_in_goal():
            return True
        elif self.timestep > self.maxTimeStep:
            return True
        else:
            return False

    def check_in_goal(self):
        is_in_goal = False
        goal_polygon = Point(self.goal[0], self.goal[1]).buffer(10)
        boat_polygon = Point(self.boat.pos[0], self.boat.pos[1]).buffer(10)
        if goal_polygon.intersects(boat_polygon):
            is_in_goal = True
        return is_in_goal

    def has_crashed(self):
        crashed = False
        for i in range(len(self.barriers)):
            barrier = np.array(self.barriers[i])
            boat = np.array(self.boat.outline)
            barrier_polygon = shapely.geometry.Polygon(barrier)
            boat_polygon = shapely.geometry.Polygon(boat)
            if barrier_polygon.intersects(boat_polygon):
                crashed = True
        return crashed

    def reset(self):
        self.__init__(self.width, self.height)

    def render(self):
        if self.count < 2:
            self.count += 1
        else:
            self.count = 0
            self.fig.clear()
            ax = self.fig.add_subplot(111)
            for p in self.barriers:
                ax.add_patch(patches.Polygon(p))
            ax.add_patch(patches.Circle(self.goal, radius=5, fc='g'))
            ax.add_patch(patches.Polygon(self.boat.outline))
            ax.set_ylim([0, self.height])
            ax.set_xlim([0, self.width])
            ax.figure.canvas.draw()
            self.fig.show()
        if self.checkTerminal():
            plt.close(self.fig)
            self.fig = plt.figure()
