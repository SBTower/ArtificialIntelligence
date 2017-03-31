import copy
import numpy as np

class Environment:

  def __init__(self, name):
    pass

  def getState(self):
    return self.state

  def getPossibleActions(self):
    pass

  def getReward(self):
    return self.reward

  def update(self, action):
    pass

  def checkTerminal(self):
    return self.done

  def reset(self):
    self.__init__()

  def enumerateState(self, state):
    return state

  def getActionSize(self):
    return len(self.getPossibleActions())

  def getStateSize(self):
    return len(self.getState())

  def render(self):
    pass

  def isActionContinuous(self):
    return False
