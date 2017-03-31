
from scipy import *
import pylab
import copy
import math
import numpy as np
import random
from Environment import Environment
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class PredictiveMaintenanceEnvironment(Environment):

  def __init__(self, name = None, numberOfAssets = 100, maxEpisodeLength = 10000):
    self.numberOfAssets = numberOfAssets
    self.initialiseAssetHealths()
    self.assetHealthDecreaseRate = np.zeros(self.numberOfAssets)
    self.maintenanceCost = 0.0
    self.count = 0
    self.maxEpisodeLength = maxEpisodeLength
    self.done = False
    self.fig = plt.figure()
    self.latestAction = None

  def initialiseAssetHealths(self):
    self.assetHealths = np.zeros(self.numberOfAssets)
    for i in range(len(self.assetHealths)):
      self.assetHealths[i] = random.uniform(0.7, 1.0)

  def getState(self):
    return self.assetHealths

  def getActionSize(self):
    return self.numberOfAssets

  def update(self, action):
    self.latestAction = action
    self.count = self.count + 1
    self.maintenanceCost = 0.0
    for i in range(len(action)):
      if action[i] > 0.5:        
        self.assetHealths[i] = min(1.0, self.assetHealths[i] + random.uniform(0.5,1.0))
        self.assetHealthDecreaseRate[i] = 0.0
        if self.maintenanceCost > 0.5:
          self.maintenanceCost = self.maintenanceCost + 0.3
        else:
          self.maintenanceCost = self.maintenanceCost + 0.1
      else:
        if self.assetHealthDecreaseRate[i] == 0.0:
          if random.uniform(0,1) < 0.005:
            self.assetHealthDecreaseRate[i] = random.uniform(0.001,0.2)
        else:
          self.assetHealthDecreaseRate[i] += random.uniform(-0.03, 0.1)
        self.assetHealths[i] = max(0.0, self.assetHealths[i] - self.assetHealthDecreaseRate[i])

  def getPossibleActions(self):
    possibleActionsMin = 0.0
    possibleActionsMax = 1.0
    minActs = []
    maxActs = []
    for _ in range(self.numberOfAssets):
      minActs.append(possibleActionsMin)
      maxActs.append(possibleActionsMax)
    possibleActions = [minActs, maxActs]
    return possibleActions

  def isActionContinuous(self):
    return True

  def getReward(self):
    reward = 0
    for health in self.assetHealths:
      if health < 0.7:
        reward = -1 * math.exp(0.7-health)
    reward = reward - self.maintenanceCost 
    return reward

  def checkTerminal(self):
    if self.count >= self.maxEpisodeLength:
      return True
    else:
      return False

  def reset(self):
    self.__init__(numberOfAssets = self.numberOfAssets, maxEpisodeLength = self.maxEpisodeLength)

  def render(self):
    if self.count % 1 == 0:
      self.fig.clear()
      ax = self.fig.add_subplot(111)
      ax.bar(range(self.numberOfAssets), self.assetHealths)
      if self.latestAction is not None:
        for i in range(len(self.latestAction)):
          if self.latestAction[i] > 0.5:
            ax.bar(i,-0.2, color='red')
      ax.set_ylim([-0.2,1])
      ax.set_xlim([0,self.numberOfAssets+1])
      ax.figure.canvas.draw()
      self.fig.show()
    if self.checkTerminal():
      plt.close(self.fig)
      self.fig = plt.figure()

