class Policy:
    def __init__(self, learner, explorer):
        self.learner = learner
        self.explorer = explorer

    def get_action(self, state):
        original_action = self.learner.get_highest_value_action(state)
        if self.explorer.explore() is True:
            action = self.explorer.get_exploratory_action(original_action)
        else:
            action = original_action
            if action is None:
                action = self.explorer.get_exploratory_action(original_action)
        return action

    def update(self, experience):
        self.learner.update(experience)
