from Agents import Agent
from Agents.History import Experience
import copy

class Controller:

  def __init__(self, environment, agent, batchSize = 1, updateTargetRate = None, render = False):
    self.env = environment
    self.agent = agent
    self.batchSize = batchSize
    self.updateTargetRate = updateTargetRate
    self.count = 0
    self.render = render

  def runOneStep(self):
    pass

  def runOneEpisode(self):
    self.env.reset()
    totalReward = 0
    while self.env.checkTerminal() is False:
      reward = self.runOneStep()
      totalReward += reward
    self.runOneStep()
    self.agent.history.resetTraces()
    self.env.reset()
    return totalReward

class BatchController(Controller):

  def runOneStep(self):
    if self.render is True:
      self.env.render()
    state = copy.copy(self.env.getState())
    action = self.agent.getAction(state)
    reward = 0.0
    if self.env.checkTerminal() is False:
      latestExperience = Experience(copy.copy(state), copy.copy(action))
      self.env.update(action)
      reward = self.env.getReward()
      latestExperience.reward = copy.copy(reward)
      state = self.env.getState()
      if self.env.checkTerminal() is False:
        action = self.agent.getAction(state)
        latestExperience.done = False
      else:
        action = 0.0
        latestExperience.done = True
      latestExperience.nextState = copy.copy(state)
      latestExperience.nextAction = copy.copy(action)
      self.agent.updateHistory(copy.copy(latestExperience))
      self.agent.updatePolicyBatch(max(1,self.batchSize))
      self.count = self.count + 1
      if self.updateTargetRate is not None:
        if self.count % self.updateTargetRate == 0:
          self.agent.policy.learner.updateTargetNetwork()
    else:
      latestExperience = Experience(copy.copy(state), copy.copy(action))
      latestExperience.reward = 0.0
      latestExperience.nextState = copy.copy(state)
      latestExperience.nextAction = 0.0
      self.agent.updateHistory(copy.copy(latestExperience))
      self.agent.updatePolicyBatch(max(1,self.batchSize))
      self.count = 0
      self.agent.history.resetTraces()
    if self.render is True:
      self.env.render()
    return reward

class OrderedController(Controller):

  def runOneStep(self):
    if self.render is True:
      self.env.render()
    state = copy.copy(self.env.getState())
    action = self.agent.getAction(state)
    reward = 0.0
    if self.env.checkTerminal() is False:
      latestExperience = Experience(copy.copy(state), copy.copy(action))
      self.env.update(action)
      reward = self.env.getReward()
      latestExperience.reward = copy.copy(reward)
      state = self.env.getState()
      if self.env.checkTerminal() is False:
        action = self.agent.getAction(state)
        latestExperience.done = False
      else:
        action = 0.0
        latestExperience.done = True
      latestExperience.nextState = copy.copy(state)
      latestExperience.nextAction = copy.copy(action)
      self.agent.updateHistory(copy.copy(latestExperience))
      self.count = self.count + 1
      if self.count % self.batchSize == 0:
        self.agent.updatePolicyOrdered(max(1,self.batchSize))
      if self.updateTargetRate is not None:
        if self.count % self.updateTargetRate == 0:
          self.count = 0
          self.agent.policy.learner.updateTargetNetwork()
    else:
      latestExperience = Experience(copy.copy(state), copy.copy(action))
      latestExperience.reward = 0.0
      latestExperience.nextState = copy.copy(state)
      latestExperience.nextAction = 0.0
      self.agent.updateHistory(copy.copy(latestExperience))
      if self.count % self.batchSize > 0:
        self.agent.updatePolicyOrdered((self.count % self.batchSize) + 1)
      self.count = 0
      self.agent.history.resetTraces()
    if self.render is True:
      self.env.render()
    return reward
      
    
      
