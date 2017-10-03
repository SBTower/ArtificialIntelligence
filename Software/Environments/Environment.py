"""Author: Stuart Tower
"""

class Environment:
    """The base Environment class. All implemented environments should extend from this and implement each of these
    fuctions
    """
    def __init__(self):
        """Initialise environment
        """
        pass

    def get_state(self):
        """Get the current state of the environment

        :return: The current state of the environment
        """
        return self.enumerate_state(self.state)

    def get_possible_actions(self):
        """Return the set of possible actions an agent can take.
        The output should either be a list of possible actions if the environment uses discrete actions, or a collection
        of ranges if the action space is continuous.
        Discrete: [a1, a2, a3, a4]
        Continuous: [[a1_min, a2_min, a3_min], [a1_max, a2_max, a3_max]]

        :return: The set of possible actions an agent can take in the environment
        """
        pass

    def get_reward(self):
        """Calculates the reward to attribute to the agent

        :return: The reward to attribute to the agent in the current state
        """
        return self.reward

    def update(self, action):
        """Update the environment by having the agent take the input action

        :param action: The action the agent is taking
        :return: None
        """
        pass

    def check_terminal(self):
        """Check if the environment is currently in a terminal state

        :return: A flag which is true if the environment is in a terminal state
        """
        return self.done

    def reset(self):
        """Reset the environment to its initial condition

        :return: None
        """
        self.__init__()

    def enumerate_state(self, state):
        """Enumerate the state representation for input to the agent policy

        :param state: The current state of the environment, represented internally
        :return: An enumerated state of the environment to feed into the agents policy
        """
        return state

    def get_action_size(self):
        """Returns the number of possible actions the agent can take

        :return: The number of possible actions
        """
        if self.is_action_continuous():
            return len(self.get_possible_actions()[0])
        else:
            return len(self.get_possible_actions())

    def get_state_size(self):
        """Return the size of the state representation

        :return: The number of values that represent the state of the environment
        """
        return len(self.get_state())

    def animate(self):
        """Update the visualisation of the environment

        :return: None
        """
        pass

    def is_action_continuous(self):
        """Check if the action space is continuous or discrete

        :return: A flag which is true if the action space is continuous
        """
        return False

    def save_figure(self):
        """Save the current visualisation as an image

        :return: None
        """
        pass
