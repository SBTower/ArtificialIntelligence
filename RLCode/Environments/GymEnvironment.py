import numpy as np
import gym
from .Environment import Environment
from gym import spaces, wrappers


class GymEnvironment(Environment):
    def __init__(self, name='CartPole-v1', render=False):
        self.name = name
        self.env = gym.make(name)
        state = self.env.reset()
        self.render = render
        self.num_figures_saved = 0
        self.num_episodes = 0
        self.state = self.enumerate_state(state)
        self.done = False

    def get_possible_actions(self):
        if type(self.env.action_space) is spaces.Discrete:
            return range(self.env.action_space.n)
        else:
            return [self.env.action_space.low, self.env.action_space.high]

    def update(self, action):
        newState, reward, done, _ = self.env.step(action)
        self.state = self.enumerate_state(newState)
        self.reward = reward
        self.done = done
        self.animate()

    def reset(self):
        state = self.env.reset()
        self.state = self.enumerate_state(state)
        self.done = False
        self.num_episodes += 1
        self.num_figures_saved = 0

    def enumerate_state(self, state):
        if type(self.env.observation_space) is spaces.Discrete:
            return np.identity(self.env.observation_space.n)[state]
        else:
            return state

    def get_action_size(self):
        if type(self.env.action_space) is spaces.Discrete:
            return self.env.action_space.n
        else:
            return len(self.env.action_space.low)

    def get_state_size(self):
        if type(self.env.observation_space) is spaces.Discrete:
            return self.env.observation_space.n
        else:
            return np.asarray(self.env.observation_space.shape)

    def animate(self):
        if self.render:
            self.env.render()

    def is_action_continuous(self):
        if type(self.env.action_space) is spaces.Discrete:
            return False
        else:
            return True
