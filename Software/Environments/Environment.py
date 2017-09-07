class Environment:
    def __init__(self):
        pass

    def get_state(self):
        return self.state

    def get_possible_actions(self):
        pass

    def get_reward(self):
        return self.reward

    def update(self, action):
        pass

    def check_terminal(self):
        return self.done

    def reset(self):
        self.__init__()

    def enumerate_state(self, state):
        return state

    def get_action_size(self):
        return len(self.get_possible_actions())

    def get_state_size(self):
        return len(self.get_state())

    def animate(self):
        pass

    def is_action_continuous(self):
        return False

    def save_figure(self, num_fig):
        pass
