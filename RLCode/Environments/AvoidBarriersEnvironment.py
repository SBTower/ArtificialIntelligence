
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

class AvoidBarriersEnvironment(Environment):

  def __init__(self, name = None, width = 200, height = 200, maxEpisodeLength = 1000):
    self.width = width
    self.height = height
    self.maxEpisodeLength = maxEpisodeLength
    pos_x = self.width/10 + random.random()*4*self.width/5
    pos_y = self.height/10 + random.random()*4*self.height/5
    orien = random.random()*2*math.pi
    self.boat = SimpleVehicle([pos_x, pos_y, orien])
    self.createBarriers()
    while self.hasCrashed() is True:
      pos_x = self.width/10 + random.random()*4*self.width/5
      pos_y = self.height/10 + random.random()*4*self.height/5
      orien = random.random()*2*math.pi
      self.boat = SimpleVehicle([pos_x, pos_y, orien])
    self.done = False
    #self.fig = plt.figure()
    self.count = 0

  def createBarriers(self):
    barrier1 = [[0, 5], [self.width, 5], [self.width, 0], [0, 0]]
    barrier2 = [[0, self.height-5], [self.width, self.height-5], [self.width, self.height], [0, self.height]]
    barrier3 = [[self.width-5, self.height], [self.width, self.height], [self.width, 0], [self.width-5, 0]]
    barrier4 = [[5, self.height], [0, self.height], [0, 0], [5, 0]]
    
    self.barriers = [barrier1, barrier2, barrier3, barrier4]

    numBarriers = random.randint(1,5)
    for i in range(numBarriers):
      randStartX = random.uniform(20, self.width-20)
      randStartY = random.uniform(20, self.height-20)
      point1 = [randStartX, randStartY]
      point2 = [point1[0]+random.uniform(15,40), point1[1]]
      point3 = [point2[0], point2[1]+random.uniform(15,40)]
      point4 = [point3[0]+random.uniform(-40,-15), point3[1]]
      self.barriers.append([point1, point2, point3, point4])

  def getState(self):
    sensorLocations = np.array(self.boat.outline)
    sensorReadings = []
    for i in range(len(sensorLocations)):

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
        sing_4 = -1

      sensorReadings.append(self.boat.sensorRange)
      sensorReadings.append(self.boat.sensorRange)
      x1 = sensorLocations[i][0]
      y1 = sensorLocations[i][1]

      for barrier in self.barriers:

        x2 = x1 + sign_1 * self.boat.sensorRange * math.cos(self.boat.pos[2])
        y2 = y1 + sign_2 * self.boat.sensorRange * math.sin(self.boat.pos[2])

        x3 = x1 + sign_3 * self.boat.sensorRange * math.sin(self.boat.pos[2])
        y3 = y1 + sign_4 * self.boat.sensorRange * math.cos(self.boat.pos[2])

        line1 = shapely.geometry.LineString([[x1, y1], [x2, y2]])
        line2 = shapely.geometry.LineString([[x1, y1], [x3, y3]])

        barrierPoly = shapely.geometry.Polygon(np.array(barrier))
        sd = self.boat.sensorRange
        while line1.intersects(barrierPoly) and sd > 0:
          sd = sd - 1
          x2 = x1 + sign_1 * sd * math.cos(self.boat.pos[2])
          y2 = y1 + sign_2 * sd * math.sin(self.boat.pos[2])
          line1 = shapely.geometry.LineString([[x1, y1], [x2, y2]])
        if sd < sensorReadings[2*i]:
          sensorReadings[2*i] = sd

        sd = self.boat.sensorRange
        while line2.intersects(barrierPoly) and sd > 0:
          sd = sd - 1
          x3 = x1 + sign_3 * sd * math.sin(self.boat.pos[2])
          y3 = y1 + sign_4 * sd * math.cos(self.boat.pos[2])
          line2 = shapely.geometry.LineString([[x1, y1], [x3, y3]])
        if sd < sensorReadings[2*i + 1]:
          sensorReadings[2*i + 1] = sd

    sensorReadings.append(copy.copy(self.boat.speed))
    sensorReadings.append(copy.copy(self.boat.angularVelocity))

    return sensorReadings

  def update(self, action):
    self.count += 1
    n1 = action%11
    n2 = action/11
    action1 = float((n1-5))/2
    action2 = float((n2-5))/200
    timestep = 0.5
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
    if self.hasCrashed():
      reward = -1000
    else:
      reward = 1
    return reward

  def checkTerminal(self):
    if self.hasCrashed():
      return True
    elif self.count > self.maxEpisodeLength:
      return True
    else:
      return False

  def hasCrashed(self):
    crashed = False
    boat = np.array(self.boat.outline)
    boatPoly = shapely.geometry.Polygon(boat)
    for barrier in self.barriers:
      barrierPoly = shapely.geometry.Polygon(barrier)
      if barrierPoly.intersects(boatPoly):
        crashed = True
    return crashed

  def reset(self):
    self.__init__(width = self.width, height = self.height, maxEpisodeLength = self.maxEpisodeLength)

  def render(self):
    if self.count % 3 == 0:
      self.fig.clear()
      ax = self.fig.add_subplot(111)
      for p in self.barriers:
        ax.add_patch(patches.Polygon(p))
      ax.add_patch(patches.Polygon(self.boat.outline))
      ax.set_ylim([0,self.height])
      ax.set_xlim([0,self.width])
      ax.figure.canvas.draw()
      self.fig.show()
    if self.checkTerminal():
      plt.close(self.fig)
      self.fig = plt.figure()










