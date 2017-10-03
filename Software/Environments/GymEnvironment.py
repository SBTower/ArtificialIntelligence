"""Author: Stuart Tower
"""

import numpy as np
import gym
from .Environment import Environment
from gym import spaces, wrappers


class GymEnvironment(Environment):
    """An environment that uses OpenAI Gym environments, formated so that they are accessible in our code base
    """
    def __init__(self, name='CartPole-v1', render=False):
        """Initialise the OpenAI Gym environment

        :param name: The name of the OpenAI Gym environment to use
        :param render: A flag specifying whether or not to render the environment to the screen
        """
        self.name = name
        self.env = gym.make(name)
        state = self.env.reset()
        self.render = render
        self.num_figures_saved = 0
        self.num_episodes = 0
        self.state = self.enumerate_state(state)
        self.done = False

    def get_possible_actions(self):
        """Get the set of possible actions available to the agent

        :return: The set of actions available to the agent
        """
        if type(self.env.action_space) is spaces.Discrete:
            return range(self.env.action_space.n)
        else:
            return [self.env.action_space.low, self.env.action_space.high]

    def update(self, action):
        """Update the environment according to the agent taking the input action

        :param action: The action taken by the agent
        :return: None
        """
        newState, reward, done, _ = self.env.step(action)
        self.state = self.enumerate_state(newState)
        self.reward = reward
        self.done = done
        self.animate()

    def reset(self):
        """Reset the environment to its initial state

        :return: None
        """
        state = self.env.reset()
        self.state = self.enumerate_state(state)
        self.done = False
        self.num_episodes += 1
        self.num_figures_saved = 0

    def enumerate_state(self, state):
        """Enumerate the state into a form to be input to the agent policy

        :param state: The internal state representation
        :return:
        """
        if type(self.env.observation_space) is spaces.Discrete:
            return np.identity(self.env.observation_space.n)[state] # One-Hot encoding
        else:
            return state

    def get_action_size(self):
        """Get the number of actions available to the agent

        :return: The size of the action space available to the agent
        """
        if type(self.env.action_space) is spaces.Discrete:
            return self.env.action_space.n
        else:
            return len(self.env.action_space.low)

    def get_state_size(self):
        """Get the size of the state space, i.e. the number of values used to represent it

        :return: The size of the state space representation
        """
        if type(self.env.observation_space) is spaces.Discrete:
            return self.env.observation_space.n
        else:
            return np.asarray(self.env.observation_space.shape)

    def animate(self):
        """Draw the environment to the screen for visualisation

        :return: None
        """
        if self.render:
            self.env.render()

    def is_action_continuous(self):
        """Check if the action space is continuous or discrete

        :return: A flag which is true if the action space is continuous
        """
        if type(self.env.action_space) is spaces.Discrete:
            return False
        else:
            return True
