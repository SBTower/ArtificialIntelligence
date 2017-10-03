"""Author: Stuart Tower
"""

from Agents.History.ExperienceHistory import ExperienceHistory


class Agent:
    """The implementation of the Agent that interacts with the environment. The agent has a history (memory) and a
    policy it uses to act on
    """
    def __init__(self, policy, number_of_planning_steps=0):
        """Initialise the parameters for the Agent

        :param policy: The policy that controls the agents actions
        :param number_of_planning_steps: The number of planning steps to use in the training
        """
        self.policy = policy
        self.history = ExperienceHistory()
        self.num_planning_steps = number_of_planning_steps

    def get_action(self, state):
        """Gets the action based on the observed state using the defined policy

        :param state: The observed state in the environment
        :return: The chosen action based on the observed state
        """
        action = self.policy.get_action(state)
        return action

    def update_policy(self, experience):
        """Update the policy based on the input experience

        :param experience: A collection of experiences to be used in the training
        :return: None
        """
        self.policy.update(experience)
        self.run_planning_steps()   # On each update run a number of planning steps

    def update_history(self, experience):
        """Update the agents history with the experience

        :param experience: An experience the agent has encountered, including states, actions and rewards
        :return: None
        """
        self.history.add_to_history([experience])

    def run_planning_steps(self):
        """Run a number of planning steps, performing a policy update for each one

        :return: None
        """
        history_for_planning = self.history.select_random_samples(self.num_planning_steps)
        for experience in history_for_planning:
            experience.next_action = self.get_action(experience.next_state)
            self.policy.update(experience)

    def update_policy_batch(self, batch_size):
        """Update the policy using a batch of randomly selected samples from the history

        :param batch_size: The number of experiences to use to update the policy
        :return: None
        """
        experience_batch = self.history.select_random_samples(batch_size)
        self.policy.update(experience_batch)

    def update_policy_ordered(self, batch_size):
        """Update the policy using an ordered sequence of samples form the history

        :param batch_size: The number of experiences to use to update the policy
        :return: None
        """
        experience_batch = self.history.select_latest_samples(batch_size)
        self.policy.update(experience_batch)
