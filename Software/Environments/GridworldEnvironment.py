from .Environment import Environment
from .Gridworld import gameEnv


class GridworldEnvironment(Environment):
    def __init__(self):
        self.env = gameEnv(partial=False, size=5)
        state = self.env.reset()
        self.state = self.enumerate_state(state)
        self.reward = 0.0
        self.done = False

    def get_possible_actions(self):
        return range(self.env.actions)

    def update(self, action):
        new_state, reward, done = self.env.step(action)
        self.state = self.enumerate_state(new_state)
        self.reward = reward
        self.done = done

    def reset(self):
        self.env.reset()
        state = self.env.reset()
        self.state = self.enumerate_state(state)
        self.done = False

    def enumerate_state(self, state):
        return state

    def get_action_size(self):
        return self.env.actions

    def get_state_size(self):
        return [84, 84, 3]
