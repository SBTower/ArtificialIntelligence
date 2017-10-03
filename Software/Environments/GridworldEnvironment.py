"""Author: Stuart Tower
"""

from .Environment import Environment
from .Gridworld import gameEnv


class GridworldEnvironment(Environment):
    """An environment that uses the Gridworld class and interfaces that environment to a format suitable for our code
    The gridworld environment is a simple game where an agent must move thorugh a grid to find the goal, and the state
    is represented using the on-screen pixels
    This example is presented in the online blog: https://medium.com/@awjuliani/simple-reinforcement-learning-with-tensorflow-part-4-deep-q-networks-and-beyond-8438a3e2b8df
    """
    def __init__(self):
        """Initialise the gridworld environment
        """
        self.env = gameEnv(partial=False, size=5)
        state = self.env.reset()
        self.state = self.enumerate_state(state)
        self.reward = 0.0
        self.done = False

    def get_possible_actions(self):
        """Get the number of possible actions for the agent

        :return: The number of possible actions
        """
        return range(self.env.actions)

    def update(self, action):
        """Update the environment with the input action

        :param action: The action the agent should take
        :return: None
        """
        new_state, reward, done = self.env.step(action)
        self.state = self.enumerate_state(new_state)
        self.reward = reward
        self.done = done

    def reset(self):
        """Reset the environment

        :return: None
        """
        self.env.reset()
        state = self.env.reset()
        self.state = self.enumerate_state(state)
        self.done = False

    def enumerate_state(self, state):
        """

        :param state: The current state of the environment
        :return: The enumerated state (which is the same as the input in this example)
        """
        return state

    def get_action_size(self):
        """Get the number of actions available to the agent

        :return: The number of possible actions
        """
        return self.env.actions

    def get_state_size(self):
        """Get the size of the gridworld

        :return: The size of the gridworld
        """
        return [84, 84, 3]
