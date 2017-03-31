import copy
import numpy as np
import gym
from Environment import Environment
from gridworld import gameEnv
from gym import spaces

class GridworldEnvironment(Environment):

  def __init__(self, name = None):
    self.env = gameEnv(partial=False, size=5)
    state = self.env.reset()
    self.state = self.enumerateState(state)
    self.done = False

  def getPossibleActions(self):
    return range(self.env.actions)

  def update(self, action):
    newState,reward,done = self.env.step(action)
    self.state = self.enumerateState(newState)
    self.reward = reward
    self.done = done

  def reset(self):
    self.env.reset()
    state = self.env.reset()
    self.state = self.enumerateState(state)
    self.done = False

  def enumerateState(self, state):
    return state

  def getActionSize(self):
    return self.env.actions

  def getStateSize(self):
    return [84,84,3]
