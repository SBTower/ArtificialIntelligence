import random

class EpsilonGreedyExplorer:

  def __init__(self, possibleActions, continuous = False, epsilon = 0.1, epsilonDecay = 0, epsilonMin = 0.1):
    self.epsilon = epsilon
    self.epsilonDecay = epsilonDecay
    self.epsilonMin = epsilonMin
    self.possibleActions = possibleActions
    self.continuous = continuous

  def explore(self):
    if self.epsilon < self.epsilonMin:
      self.epsilon = self.epsilonMin
    if random.random() < self.epsilon:
      return True
    return False

  def getExploratoryAction(self, originalAction):
    self.epsilon = self.epsilon * (1 - self.epsilonDecay)
    if self.continuous is False:
      return random.choice(self.possibleActions)
    else:
      actions = [random.uniform(self.possibleActions[0][i],self.possibleActions[1][i]) for i in range(len(self.possibleActions[0]))]
      return actions
