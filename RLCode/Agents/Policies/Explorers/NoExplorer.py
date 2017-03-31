import random

class NoExplorer:

  def __init__(self, possibleActions, continuous = False, epsilon = 0.1, epsilonDecay = 0, epsilonMin = 0.1):
    self.epsilon = epsilon
    self.epsilonDecay = epsilonDecay
    self.epsilonMin = epsilonMin
    self.possibleActions = possibleActions
    self.continuous = continuous

  def explore(self):
    return False

  def getExploratoryAction(self, originalAction):
    return originalAction
