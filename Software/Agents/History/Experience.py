"""Author: Stuart Tower
"""

import numpy as np


class Experience:
    """A snapshot of an agents experience to be saved in the history. An experience contains the initial state, the
    action taken, the reward received, the next state and the next action taken.

    """
    def __init__(self, state=None, action=None, reward=None, next_state=None, next_action=None, done=False):
        """Initialise the experience with the appropriate values

        :param state: The initial state
        :param action: The action taken
        :param reward: The reward obtained
        :param next_state: The resultant state
        :param next_action: The action taken in the resultant state
        :param done: Whether the resultant state is terminal
        """
        self.state = state
        self.action = action
        self.reward = reward
        self.nextState = next_state
        self.nextAction = next_action
        self.done = done

    def equals(self, experience):
        """Checks if the input experience is equal to this experience

        :param experience: A different experience to compare to this one
        :return: Boolean which is true if the two experiences are equivalent
        """
        if np.array_equal(self.state, experience.state)\
                and np.array_equal(self.action, experience.action)\
                and self.reward == experience.reward\
                and np.array_equal(self.nextState, experience.nextState):
            return True
        return False
