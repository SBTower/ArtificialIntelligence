import numpy as np

class Policy:

  def __init__(self, learner, explorer):
    self.learner = learner
    self.explorer = explorer

  def getAction(self, state):
    originalAction = self.learner.getHighestValueAction(state)
    if self.explorer.explore() is True:     
      action = self.explorer.getExploratoryAction(originalAction)
    else:
      action = originalAction
      if action is None:
        action = self.explorer.getExploratoryAction(originalAction)
    return action

  def update(self, experience, ordered):
    self.learner.update(experience, ordered)

