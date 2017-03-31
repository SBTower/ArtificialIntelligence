from Experience import Experience
import random
import copy

class ExperienceHistory:

  def __init__(self, maxSize = 10000):
    self.history = []
    self.maxSize = maxSize

  def addToHistory(self, batch):
    for experience in batch:
      index = self.findInHistory(experience)
      if index is None:
        self.history.append(experience)
      else:
        self.history.pop(index)
        self.history.append(experience)
    self.removeOldExperience()

  def removeOldExperience(self):
    while len(self.history) > self.maxSize:
      self.history.pop(0)

  def selectRandomSamples(self, numSamples):
    samples = []
    numSamples = min(numSamples, len(self.history))
    for n in range(numSamples):
      samples.append(self.selectRandomSample())
    return samples

  def selectRandomSample(self):
    sample = None
    if len(self.history) > 0:
      index = random.randint(0, len(self.history) - 1)
      sample = copy.copy(self.history[index])
    return sample

  def getLatestExperience(self):
    if len(self.history) > 0:
      index = len(self.history) - 1
      sample = self.history[index]
    else:
      sample = None
    return sample

  def selectLatestSamples(self, n):
    if len(self.history) > 0:
      index1 = max(len(self.history)-n,0)
      index2 = len(self.history)
      samples = self.history[index1:index2]
    else:
      samples = [None]
    return samples

  def clearHistory(self):
    self.history = []

  def findInHistory(self, experience):
    index = None
    for i in range(len(self.history)):
      if self.history[i].equals(experience):
        index = i
        break
    return index

  def updateTraces(self, experience, lmda):
    self.addToHistory([experience])
    for exp in self.history:
      if exp.equals(experience):
        exp.trace = 1
      else:
        exp.trace = exp.trace * lmda

  def resetTraces(self):
    for exp in self.history:
      exp.trace = 0
