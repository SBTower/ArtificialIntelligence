"""Author: Stuart Tower
"""

from Agents.History.Experience import Experience
import copy


class Controller:
    """The controller class handles the running of an episode, interfacing the agent and the environment
    """
    def __init__(self, environment, agent, batch_size=1, update_target_rate=None):
        """Initialise the controller

        :param environment: The environment to train on
        :param agent: The agent acting in the environment
        :param batch_size: The size of the batch to use in each training step
        :param update_target_rate: The rate at which the target network is updated (if there is any)
        """
        self.env = environment
        self.agent = agent
        self.batch_size = batch_size
        self.update_target_rate = update_target_rate
        self.count = 0

    def run_one_step(self):
        """Run one time step of the environment

        :return: The reward gained in the time step
        """
        pass

    def run_one_episode(self):
        """Run an entire episode of the environment.
        This runs until the environment enters a terminal state, before resetting the environment

        :return: The total reward of the episode
        """
        total_reward = 0
        while self.env.check_terminal() is False:
            reward = self.run_one_step()
            total_reward += reward
        self.run_one_step()
        self.env.reset()
        return total_reward


class BatchController(Controller):
    """The BatchController runs each training step updating the policy using a random batch of previous experience
    """
    def run_one_step(self):
        """Perform one iteration through the environment

        :return: The reward gained in the time step
        """
        # Get the current state, action and initialise the reward
        state = copy.copy(self.env.get_state())
        action = self.agent.get_action(state)
        reward = 0.0
        # Check if the environment has reached a terminal state
        if self.env.check_terminal() is False:
            # Save the initial state and action to an 'experience'
            latest_experience = Experience(copy.copy(state), copy.copy(action))
            # Update the environment using the chosne action
            self.env.update(action)
            # Get the reward to attribute to the agent and save to the experience to save
            reward = self.env.get_reward()
            latest_experience.reward = copy.copy(reward)
            # Get the updated state
            state = self.env.get_state()
            if self.env.check_terminal() is False:
                # If the new state isn't terminal, save the next action and the 'done' flag to the experience
                action = self.agent.get_action(state)
                latest_experience.done = False
            else:
                # If the new state is terminal, save a dummy action and the 'done' flag to the experience
                action = 0.0
                latest_experience.done = True
            latest_experience.next_state = copy.copy(state)
            latest_experience.next_action = copy.copy(action)
            # Update the history with the latest experience
            self.agent.update_history(copy.copy(latest_experience))
            # Update the agents policy using a batch of experiences chosen from the history
            self.agent.update_policy_batch(max(1, self.batch_size))
            self.count += 1
            # Update the target network if appropriate
            if self.update_target_rate is not None:
                if self.count % self.update_target_rate == 0:
                    self.agent.policy.learner.update_target_network()
        else:
            # If the environment is in a terminal state, record this and perform a policy update
            latest_experience = Experience(copy.copy(state), copy.copy(action))
            latest_experience.reward = 0.0
            latest_experience.next_state = copy.copy(state)
            latest_experience.next_action = 0.0
            latest_experience.done = True
            self.agent.update_history(copy.copy(latest_experience))
            self.agent.update_policy_batch(max(1, self.batch_size))
            self.count = 0
        return reward


class OrderedController(Controller):
    """The Ordered Controller runs a time step performing a policy update using the latest experience in order
    """
    def run_one_step(self):
        """Perform one iteration through the environment

        :return: The reward gained in the time step
        """
        state = copy.copy(self.env.get_state())
        action = self.agent.get_action(state)
        reward = 0.0
        if self.env.check_terminal() is False:
            latest_experience = Experience(copy.copy(state), copy.copy(action))
            self.env.update(action)
            reward = self.env.get_reward()
            latest_experience.reward = copy.copy(reward)
            state = self.env.get_state()
            if self.env.check_terminal() is False:
                action = self.agent.get_action(state)
                latest_experience.done = False
            else:
                action = 0.0
                latest_experience.done = True
            latest_experience.next_state = copy.copy(state)
            latest_experience.next_action = copy.copy(action)
            self.agent.update_history(copy.copy(latest_experience))
            self.count += 1
            # If the latest history has a large enough batch, perform an update
            # CHECK IF THIS IS THE RIGHT METHOD
            if self.count % self.batch_size == 0:
                self.agent.update_policy_ordered(max(1, self.batch_size))
            if self.update_target_rate is not None:
                if self.count % self.update_target_rate == 0:
                    self.count = 0
                    self.agent.policy.learner.update_target_network()
        else:
            latest_experience = Experience(copy.copy(state), copy.copy(action))
            latest_experience.reward = 0.0
            latest_experience.next_state = copy.copy(state)
            latest_experience.next_action = 0.0
            latest_experience.done = True
            self.agent.update_history(copy.copy(latest_experience))
            # Perform an update on all of the previous experiences that haven't been updated
            if self.count % self.batch_size > 0:
                self.agent.update_policy_ordered((self.count % self.batch_size) + 1)
            self.count = 0
        return reward
