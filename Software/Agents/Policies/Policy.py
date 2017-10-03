"""Author: Stuart Tower
"""

class Policy:
    """Handles how to select an action, and how to update the current policy

    Formed of a learner and an explorer. The explorer controls when and how to explore the environment. The learner
    contains the current policy and controls how to update the policy based on experience
    """
    def __init__(self, learner, explorer):
        """Initialise the policy

        :param learner: The learner to use in the policy
        :param explorer: The explorer to use in the policy
        """
        self.learner = learner
        self.explorer = explorer

    def get_action(self, state, action_diff=0., action_scale=1.):
        """Gets the action to feed into the environment, either the current best action from the learner, or an
        exploratory action from the explorer. Note that the action is scaled based on the requirements of the
        environment and the expected output bounds from the learner. The action is value is shifted and then scaled.

        :param state: The observed state of the environment
        :param action_diff: The amount to shift the output action to scale it appropriately
        :param action_scale: The amount to scale the output action to scale it appropriately
        :return: The action for the agent to perform on the environment
        """
        original_action = self.learner.get_highest_value_action(state)
        if self.explorer.explore() is True:
            action = self.explorer.get_exploratory_action(original_action)
        else:
            action = original_action
            if action is None:
                action = self.explorer.get_exploratory_action(original_action)
            else:
                action = action_scale * (action - action_diff)
        return action

    def update(self, experience):
        """Update the current policy based on the experience

        :param experience: A collection of previous experiences with which to update the current policy
        :return: None
        """
        self.learner.update(experience)
