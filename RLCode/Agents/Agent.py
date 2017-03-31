from History import ExperienceHistory
from Policies import Policy

class Agent:

  def __init__(self, policy, numPlanningSteps = 0, lmda = 0, minimumWeight = 0.01):
    self.policy = policy
    self.history = ExperienceHistory()
    self.numPlanningSteps = numPlanningSteps
    self.lmda = lmda
    self.minimumWeight = minimumWeight

  def getAction(self, state):
    action = self.policy.getAction(state)
    return action

  def updatePolicy(self, experience):
    if self.lmda > 0:
      for exp in self.history.history:
        exp.reward = experience.reward
        exp.nextState = experience.nextState
        exp.nextAction = experience.nextAction
        if exp.trace > self.minimumWeight:
          self.policy.update(exp)
    else:
      self.policy.update(experience)
    self.runPlanningSteps()

  def updateHistory(self, experience):
    self.history.addToHistory([experience])
    self.history.updateTraces(experience, self.lmda)

  def runPlanningSteps(self):
    historyForPlanning = self.history.selectRandomSamples(self.numPlanningSteps)
    for experience in historyForPlanning:
      experience.nextAction = self.getAction(experience.nextState)
      experience.trace = 1
      self.policy.update(experience)

  def updatePolicyBatch(self, batchSize):
    experienceBatch = self.history.selectRandomSamples(batchSize)
    self.policy.update(experienceBatch, False)

  def updatePolicyOrdered(self, batchSize):
    experienceBatch = self.history.selectLatestSamples(batchSize)
    self.policy.update(experienceBatch, True)
