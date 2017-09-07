import numpy as np


class Experience:
    def __init__(self, state=None, action=None, reward=None, next_state=None, next_action=None, done=False):
        self.state = state
        self.action = action
        self.reward = reward
        self.nextState = next_state
        self.nextAction = next_action
        self.done = done

    def equals(self, experience):
        if np.array_equal(self.state, experience.state)\
                and np.array_equal(self.action, experience.action)\
                and self.reward == experience.reward\
                and np.array_equal(self.nextState, experience.nextState):
            return True
        return False
