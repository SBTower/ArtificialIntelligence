from scipy import *
import math
import random
from .Vehicles.SimpleVehicle import SimpleVehicle
from .Environment import Environment
from shapely.geometry import Point

import matplotlib.pyplot as plt
import matplotlib.patches as patches


class NavigateToGoalEnvironment(Environment):
    def __init__(self, width=200, height=200, max_episode_length=100, render=False):
        self.width = width
        self.height = height
        self.maxEpisodeLength = max_episode_length
        self.render = render
        self.num_figures_saved = 0
        self.num_episodes = 0
        self.current_episode_length = 0
        pos_x = self.width / 10 + random.random() * 4 * self.width / 5
        pos_y = self.height / 10 + random.random() * 4 * self.height / 5
        orientation = random.random() * 2 * math.pi
        self.boat = SimpleVehicle([pos_x, pos_y, orientation])
        self.goal = self.create_goal()
        self.fig = plt.figure()
        self.initialise_display()
        if self.render is True:
            self.save_figure()
            self.animate()

    def get_state(self):
        sensor_readings = []

        #    sensor_readings.append(copy.copy(self.boat.pos[0]))
        #    sensor_readings.append(copy.copy(self.boat.pos[1]))
        #    sensor_readings.append(copy.copy(self.boat.pos[2]))
        #    sensor_readings.append(copy.copy(self.goal[0]))
        #    sensor_readings.append(copy.copy(self.goal[1]))
        sensor_readings.append(self.goal[0] - self.boat.pos[0])
        sensor_readings.append(self.goal[1] - self.boat.pos[1])
        sensor_readings.append(self.boat.pos[2])
        #    sensor_readings.append(copy.copy(self.boat.speed))
        #    sensor_readings.append(copy.copy(self.boat.angularVelocity))
        #    sensor_readings.append(self.getDistanceToGoal())
        #    sensor_readings.append(self.getAngleToGoal())

        return sensor_readings

    def create_goal(self):
        goal_x = self.width / 10 + random.random() * 4 * self.width / 5
        goal_y = self.height / 10 + random.random() * 4 * self.height / 5
        goal = [goal_x, goal_y]
        return goal

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
        self.current_episode_length += 1
        n1 = action % 11
        n2 = action / 11
        action1 = float((n1 - 5)) / 2
        action2 = float((n2 - 5)) / 200
        timestep = 1
        self.boat.change_acceleration(action1)
        self.boat.change_angular_acceleration(action2)
        self.boat.update_position(timestep)
        self.animate()

    def get_possible_actions(self):
        possible_actions = range(121)
        return possible_actions

    def get_reward(self):
        if self.check_in_goal():
            reward = 0
        else:
            reward = -1  # math.exp((50.0 - self.getDistanceToGoal())/50.0)/math.exp(1.0)
        return reward

    def check_terminal(self):
        if self.current_episode_length > self.maxEpisodeLength:
            return True
        elif self.check_in_goal():
            return True
        else:
            return False

    def check_in_goal(self):
        in_goal = False
        goal_polygon = Point(self.goal[0], self.goal[1]).buffer(10)
        boat_polygon = Point(self.boat.pos[0], self.boat.pos[1]).buffer(10)
        if goal_polygon.intersects(boat_polygon):
            in_goal = True
        return in_goal

    def reset(self):
        self.current_episode_length = 0
        pos_x = self.width / 10 + random.random() * 4 * self.width / 5
        pos_y = self.height / 10 + random.random() * 4 * self.height / 5
        orientation = random.random() * 2 * math.pi
        self.boat.reset_position([pos_x, pos_y, orientation])
        self.goal = self.create_goal()
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
        self.boat_patch = patches.Polygon(self.boat.outline)
        self.ax.add_patch(self.boat_patch)
        self.goal_patch = patches.Circle(self.goal, radius=5, fc='g')
        self.ax.add_patch(self.goal_patch)
        self.ax.set_ylim([0, self.height])
        self.ax.set_xlim([0, self.width])
        self.ax.figure.canvas.draw()

    def animate(self):
        if self.render:
            self.boat_patch.set_xy(self.boat.outline)
            self.goal_patch.center = self.goal
            self.ax.figure.canvas.draw()
            self.fig.show()
            self.save_figure()

    def save_figure(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        dir_path += '/Videos/NavToGoalEnv/Episode-' + str(self.num_episodes)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        self.fig.savefig(dir_path + '/Image_' + str(self.num_figures_saved).zfill(5))
        self.num_figures_saved += 1
