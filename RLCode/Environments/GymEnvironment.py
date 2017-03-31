import copy
import numpy as np
import gym
from Environment import Environment
from gym import spaces
import cv2

class GymEnvironment(Environment):

  def __init__(self, name = 'CartPole-v0'):
    self.name = name
    self.env = gym.make(name)
    state = self.env.reset()
    self.state = self.enumerateState(state)
    self.done = False

  def getPossibleActions(self):
    if type(self.env.action_space) is spaces.Discrete:
      return range(self.env.action_space.n)
    else:
      return [self.env.action_space.low, self.env.action_space.high]

  def update(self, action):
    newState,reward,done,_ = self.env.step(action)
    self.state = self.enumerateState(newState)
    self.reward = reward
    self.done = done

  def reset(self):
    self.env.reset()
    state = self.env.reset()
    self.state = self.enumerateState(state)
    self.done = False

  def enumerateState(self, state):
    if type(self.env.observation_space) is spaces.Discrete:
      return np.identity(self.env.observation_space.n)[state]
    else:
      return state

  def getActionSize(self):
    if type(self.env.action_space) is spaces.Discrete:
      return self.env.action_space.n
    else:
      return len(self.env.action_space.low)

  def getStateSize(self):
    if type(self.env.observation_space) is spaces.Discrete:
      return self.env.observation_space.n
    else:
      return self.env.observation_space.shape

  def render(self):
    self.env.render()

  def isActionContinuous(self):
    if type(self.env.action_space) is spaces.Discrete:
      return False
    else:
      return True
