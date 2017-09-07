from Agents.History.ExperienceHistory import ExperienceHistory


class Agent:
    def __init__(self, policy, number_of_planning_steps=0):
        self.policy = policy
        self.history = ExperienceHistory()
        self.num_planning_steps = number_of_planning_steps

    def get_action(self, state):
        action = self.policy.get_action(state)
        return action

    def update_policy(self, experience):
        self.policy.update(experience)
        self.run_planning_steps()

    def update_history(self, experience):
        self.history.add_to_history([experience])

    def run_planning_steps(self):
        history_for_planning = self.history.select_random_samples(self.num_planning_steps)
        for experience in history_for_planning:
            experience.next_action = self.get_action(experience.next_state)
            self.policy.update(experience)

    def update_policy_batch(self, batch_size):
        experience_batch = self.history.select_random_samples(batch_size)
        self.policy.update(experience_batch)

    def update_policy_ordered(self, batch_size):
        experience_batch = self.history.select_latest_samples(batch_size)
        self.policy.update(experience_batch)
