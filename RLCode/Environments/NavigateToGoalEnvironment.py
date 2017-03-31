
from scipy import *
import pylab
import copy
import math
import numpy as np
import random
from Vehicles.SimpleVehicle import SimpleVehicle
from Environment import Environment
import shapely.geometry
from shapely.geometry import Point
from shapely.geometry import LineString

import matplotlib.pyplot as plt
import matplotlib.patches as patches

class NavigateToGoalEnvironment(Environment):

  def __init__(self, name = None, width = 200, height = 200, maxEpisodeLength = 1000):
    self.width = width
    self.height = height
    self.maxEpisodeLength = maxEpisodeLength
    pos_x = self.width/10 + random.random()*4*self.width/5
    pos_y = self.height/10 + random.random()*4*self.height/5
    orien = random.random()*2*math.pi
    self.boat = SimpleVehicle([pos_x, pos_y, orien])
    self.createGoal()
    self.done = False
    #self.fig = plt.figure()
    self.count = 0

  def getState(self):
    sensorReadings = []

#    sensorReadings.append(copy.copy(self.boat.pos[0]))
#    sensorReadings.append(copy.copy(self.boat.pos[1]))
#    sensorReadings.append(copy.copy(self.boat.pos[2]))
#    sensorReadings.append(copy.copy(self.goal[0]))
#    sensorReadings.append(copy.copy(self.goal[1]))
    sensorReadings.append(self.goal[0] - self.boat.pos[0])
    sensorReadings.append(self.goal[1] - self.boat.pos[1])
    sensorReadings.append(self.boat.pos[2])
#    sensorReadings.append(copy.copy(self.boat.speed))
#    sensorReadings.append(copy.copy(self.boat.angularVelocity))
#    sensorReadings.append(self.getDistanceToGoal())
#    sensorReadings.append(self.getAngleToGoal())

    return sensorReadings    

  def createGoal(self):
    goal_x = self.width/10 + random.random()*4*self.width/5
    goal_y = self.height/10 + random.random()*4*self.height/5
    self.goal = [goal_x, goal_y]

  def getDistanceToGoal(self):
    dist_x = self.goal[0] - self.boat.pos[0]
    dist_y = self.goal[1] - self.boat.pos[1]
    distance = math.sqrt(dist_x*dist_x + dist_y*dist_y)
    return distance

  def getAngleToGoal(self):
    dist_x = self.goal[0] - self.boat.pos[0]
    dist_y = self.goal[1] - self.boat.pos[1]

    if dist_y == 0:
      if dist_x == 0:
        angleToGoal = 0
      elif dist_x > 0:
        angleToGoal = math.pi/2
      else:
        angleToGoal = 3*math.pi/2
    elif dist_y > 0:
      if dist_x >= 0:
        angleToGoal = math.atan(abs(dist_x)/abs(dist_y))
      else:
        angleToGoal = 2*math.pi - math.atan(abs(dist_x)/abs(dist_y))
    else:
      if dist_x >= 0:
        angleToGoal = math.pi - math.atan(abs(dist_x)/abs(dist_y))
      else:
        angleToGoal = math.pi + math.atan(abs(dist_x)/abs(dist_y))
    return angleToGoal

  def update(self, action):
    self.count += 1
    n1 = action%11
    n2 = action/11
    action1 = float((n1-5))/2
    action2 = float((n2-5))/200
    timestep = 1
    self.boat.changeAcceleration(action1)
    self.boat.changeAngularAcceleration(action2)
    self.boat.updatePosition(timestep)
    if self.checkTerminal():
      self.done = True
    else:
      self.done = False

  def getPossibleActions(self):
    possibleActions = range(121)
    return possibleActions

  def getReward(self):
    if self.checkInGoal():
      reward = 0
    else:
      reward = -1#math.exp((50.0 - self.getDistanceToGoal())/50.0)/math.exp(1.0)
    return reward

  def checkTerminal(self):
    if self.count > self.maxEpisodeLength:
      return True
    elif self.checkInGoal():
      return True
    else:
      return False

  def checkInGoal(self):
    inGoal = False
    goalPoly = Point(self.goal[0], self.goal[1]).buffer(10)
    boatPoly = Point(self.boat.pos[0], self.boat.pos[1]).buffer(10)
    if goalPoly.intersects(boatPoly):
      inGoal = True
    return inGoal

  def reset(self):
    self.__init__(width = self.width, height = self.height, maxEpisodeLength = self.maxEpisodeLength)

  def render(self):
    if self.count % 2 == 0:
      self.fig.clear()
      ax = self.fig.add_subplot(111)
      ax.add_patch(patches.Circle(self.goal, radius = 5, fc='g'))
      ax.add_patch(patches.Polygon(self.boat.outline))
      ax.plot([self.boat.outline[1][0], self.boat.outline[2][0]], [self.boat.outline[1][1], self.boat.outline[2][1]], color='red', linewidth=2)
      ax.set_ylim([0,self.height])
      ax.set_xlim([0,self.width])
      ax.figure.canvas.draw()
      self.fig.show()
    if self.checkTerminal():
      plt.close(self.fig)
      self.fig = plt.figure()










