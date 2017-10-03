"""Author: Stuart Tower
"""

from scipy import *
import math
import random
from .Vehicles.SimpleVehicle import SimpleVehicle
from .Environment import Environment
from shapely.geometry import Point

import matplotlib.pyplot as plt
import matplotlib.patches as patches


class NavigateToGoalEnvironment(Environment):
    """An environment describing an autonomous vehicle navigating to a defined waypoint
    """
    def __init__(self, width=200, height=200, max_episode_length=100, render=False):
        """

        :param width: The width of the playground the autonomous vehicle can move through
        :param height: The height of the playground the autonomous vehicle can move through
        :param max_episode_length: The maximum length of a single episode in time steps
        :param render: A flag describing whether or not to render the environment to the screen
        """
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
        """Get the state of the environment
        The state representation is a combination of position and velocity of the boat and goal.

        :return: The current state
        """
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
        """Create the target waypoint for the autonomous vehicle

        :return: The location of the goal (waypoint) the vehicle has to navigate towards
        """
        goal_x = self.width / 10 + random.random() * 4 * self.width / 5
        goal_y = self.height / 10 + random.random() * 4 * self.height / 5
        goal = [goal_x, goal_y]
        return goal

    def get_distance_to_goal(self):
        """Get the current straight line distance to the goal

        :return: The current distance to the goal
        """
        dist_x = self.goal[0] - self.boat.pos[0]
        dist_y = self.goal[1] - self.boat.pos[1]
        distance = math.sqrt(dist_x * dist_x + dist_y * dist_y)
        return distance

    def get_angle_to_goal(self):
        """Get the current angle between the boats heading and the heading towards the goal

        :return: The current bearing to the goal
        """
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
        """Update the environment with the agent taking the input action

        :param action: The action taken by the agent
        :return: None
        """
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
        """Get the set of possible actions available to the agent

        :return: The set of p[ossible actions available to the agent
        """
        possible_actions = range(121)
        return possible_actions

    def get_reward(self):
        """Get the reward to attribute to the agent based on the current state

        :return: The reward to attribute to the agent
        """
        if self.check_in_goal():
            reward = 0
        else:
            reward = -1  # math.exp((50.0 - self.getDistanceToGoal())/50.0)/math.exp(1.0)
        return reward

    def check_terminal(self):
        """Check if the environment is in a terminal state
        The state is terminal if the episode is longer than the allowable limit, or if the agent is at the goal

        :return: A flag, true if the environment is in a terminal state
        """
        if self.current_episode_length > self.maxEpisodeLength:
            return True
        elif self.check_in_goal():
            return True
        else:
            return False

    def check_in_goal(self):
        """Check if the agent is currently in the goal

        :return: A flag, true if the agent is currently in the goal
        """
        in_goal = False
        goal_polygon = Point(self.goal[0], self.goal[1]).buffer(10)
        boat_polygon = Point(self.boat.pos[0], self.boat.pos[1]).buffer(10)
        if goal_polygon.intersects(boat_polygon):
            in_goal = True
        return in_goal

    def reset(self):
        """Reset the environment to its initial state

        :return: None
        """
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
        """Initialise the display for drawing the environment to the screen

        :return: None
        """
        self.ax = self.fig.add_subplot(111)
        self.boat_patch = patches.Polygon(self.boat.outline)
        self.ax.add_patch(self.boat_patch)
        self.goal_patch = patches.Circle(self.goal, radius=5, fc='g')
        self.ax.add_patch(self.goal_patch)
        self.ax.set_ylim([0, self.height])
        self.ax.set_xlim([0, self.width])
        self.ax.figure.canvas.draw()

    def animate(self):
        """Update the visualisation of the environment

        :return: None
        """
        if self.render:
            self.boat_patch.set_xy(self.boat.outline)
            self.goal_patch.center = self.goal
            self.ax.figure.canvas.draw()
            self.fig.show()
            self.save_figure()

    def save_figure(self):
        """Save the current visualisation of the environment to an image

        :return: None
        """
        dir_path = os.path.dirname(os.path.realpath(__file__))
        dir_path += '/Videos/NavToGoalEnv/Episode-' + str(self.num_episodes)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        self.fig.savefig(dir_path + '/Image_' + str(self.num_figures_saved).zfill(5))
        self.num_figures_saved += 1
