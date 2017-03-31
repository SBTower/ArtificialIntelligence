import copy
import numpy as np
from Environment import Environment

import matplotlib.pyplot as plt
import matplotlib.patches as patches

class MazeEnvironment(Environment):

  def __init__(self, name='Maze1'):
    self.name = name
    self.createMaze(self.name)
    self.start = [1,1]
    self.goal = [9,11]
    self.state = copy.copy(self.start)
    #self.fig = plt.figure()

  def getState(self):
    return self.enumerateState(self.state)

  def setState(self, state):
    self.state = state

  def createMaze(self, name):
    if name is "Maze1":
      self.maze = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
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
      self.maze = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
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
      print "Unknown Maze. Loading Default"
      self.createMaze("Maze1")

  def loadMaze(self, name):
    self.__init__(name)

  def getReward(self):
    if self.checkTerminal():
      reward = 0
    else:
      reward = -1
    return reward

  def update(self, action):
    prevPosition = copy.copy(self.state)
    if action == 0:
      self.state[0] = self.state[0] - 1
    elif action == 1:
      self.state[1] = self.state[1] - 1
    elif action == 2:
      self.state[1] = self.state[1] + 1
    elif action == 3:
      self.state[0] = self.state[0] + 1
    if self.maze[self.state[0]][self.state[1]] == 1:
      self.state = prevPosition

  def enumerateState(self, state):
    n = len(self.maze)*len(self.maze[0])
    return np.identity(n)[state[0]*len(self.maze) + state[1]]

  def checkTerminal(self):
    if self.state == self.goal:
      return True
    return False

  def reset(self):
    self.__init__(self.name)

  def getPossibleActions(self):
    return [0,1,2,3]

  def render(self):
    self.fig.clear()
    ax = self.fig.add_subplot(111)
    for row in range(len(self.maze)):
      for column in range(len(self.maze[0])):
        if [row, column] == self.state:
          ax.add_patch(patches.Rectangle((column, row), 1, 1, facecolor="red"))
        elif [row, column] == self.goal:
          ax.add_patch(patches.Rectangle((column, row), 1, 1, facecolor="green"))
        elif self.maze[row][column] == 1:
          ax.add_patch(patches.Rectangle((column, row), 1, 1, facecolor="black"))
    ax.autoscale()
    ax.figure.canvas.draw()
    self.fig.show()
    if self.checkTerminal():
      plt.close(self.fig)
      self.fig = plt.figure()
