import random
import copy


class ExperienceHistory:
    def __init__(self, max_size=10000):
        self.history = []
        self.maxSize = max_size

    def add_to_history(self, batch):
        for experience in batch:
            index = self.find_in_history(experience)
            if index is None:
                self.history.append(experience)
            else:
                self.history.pop(index)
                self.history.append(experience)
        self.remove_old_experience()

    def remove_old_experience(self):
        while len(self.history) > self.maxSize:
            self.history.pop(0)

    def select_random_samples(self, number_of_samples):
        samples = []
        number_of_samples = min(number_of_samples, len(self.history))
        for n in range(number_of_samples):
            samples.append(self.select_random_sample())
        return samples

    def select_random_sample(self):
        sample = None
        if len(self.history) > 0:
            index = random.randint(0, len(self.history) - 1)
            sample = copy.copy(self.history[index])
        return sample

    def get_latest_experience(self):
        if len(self.history) > 0:
            index = len(self.history) - 1
            sample = self.history[index]
        else:
            sample = None
        return sample

    def select_latest_samples(self, n):
        if len(self.history) > 0:
            index1 = max(len(self.history) - n, 0)
            index2 = len(self.history)
            samples = self.history[index1:index2]
        else:
            samples = [None]
        return samples

    def clear_history(self):
        self.history = []

    def find_in_history(self, experience):
        index = None
        for i in range(len(self.history)):
            if self.history[i].equals(experience):
                index = i
                break
        return index
