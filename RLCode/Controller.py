from Agents.History.Experience import Experience
import copy


class Controller:
    def __init__(self, environment, agent, batch_size=1, update_target_rate=None):
        self.env = environment
        self.agent = agent
        self.batch_size = batch_size
        self.update_target_rate = update_target_rate
        self.count = 0

    def run_one_step(self):
        pass

    def run_one_episode(self):
        total_reward = 0
        while self.env.check_terminal() is False:
            reward = self.run_one_step()
            total_reward += reward
        self.run_one_step()
        self.env.reset()
        return total_reward


class BatchController(Controller):
    def run_one_step(self):
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
            self.agent.update_policy_batch(max(1, self.batch_size))
            self.count += 1
            if self.update_target_rate is not None:
                if self.count % self.update_target_rate == 0:
                    self.agent.policy.learner.update_target_network()
        else:
            latest_experience = Experience(copy.copy(state), copy.copy(action))
            latest_experience.reward = 0.0
            latest_experience.next_state = copy.copy(state)
            latest_experience.next_action = 0.0
            self.agent.update_history(copy.copy(latest_experience))
            self.agent.update_policy_batch(max(1, self.batch_size))
            self.count = 0
        return reward


class OrderedController(Controller):
    def run_one_step(self):
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
            self.agent.update_history(copy.copy(latest_experience))
            if self.count % self.batch_size > 0:
                self.agent.update_policy_ordered((self.count % self.batch_size) + 1)
            self.count = 0
        return reward
