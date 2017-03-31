import random
import numpy as np

class NoisyActionExplorer:

  def __init__(self, possibleActions, continuous = False, epsilon = 0.1, epsilonDecay = 0., epsilonMin = 0.):
    self.possibleActions = possibleActions
    self.continuous = continuous
    self.epsilon = epsilon
    self.epsilonDecay = epsilonDecay
    self.epsilonMin = epsilonMin

  def explore(self):
    if self.epsilon < self.epsilonMin:
      return False
    else:
      return True

  def getExploratoryAction(self, originalAction):
    self.epsilon = self.epsilon * (1 - self.epsilonDecay)
    if self.continuous is False:
      action = int(round(originalAction + np.random.normal(0, self.epsilon*len(self.possibleActions)/5)))
      if action < min(self.possibleActions):
        action = min(self.possibleActions)
      if action > max(self.possibleActions):
        action = max(self.possibleActions)
    else:
      action = []
      for i in range(len(originalAction)):
        new_act = originalAction[i] + np.random.normal(0, self.epsilon)
        if new_act > self.possibleActions[1][i]:
          new_act = self.possibleActions[1][i]
        elif new_act < self.possibleActions[0][i]:
          new_act = self.possibleActions[0][i]
        action.append(new_act)
    return action
