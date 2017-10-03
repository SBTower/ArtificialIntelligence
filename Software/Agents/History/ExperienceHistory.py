"""Author: Stuart Tower
"""

import random
import copy


class ExperienceHistory:
    """A collection of experiences representing an agents history of interactions with an environment
    """
    def __init__(self, max_size=10000):
        """Initialise the history

        :param max_size: The maximum size of the history
        """
        self.history = []
        self.maxSize = max_size

    def add_to_history(self, batch):
        """Adds a collection of experiences to the history. If the size of the history is larger than the limit then
        the oldest experiences are removed from the history until the size of the history is allowed.

        :param batch: The collection of experiences to add to the history
        :return:
        """
        for experience in batch:
            index = self.find_in_history(experience)    # Check if experience is already in the history
            if index is None:
                self.history.append(experience)
            else:
                self.history.pop(index)
                self.history.append(experience)
        self.remove_old_experience()

    def remove_old_experience(self):
        """Remove the oldest experiences from the history if the size of the history is larger than the allowable limit

        :return: None
        """
        while len(self.history) > self.maxSize:
            self.history.pop(0)

    def select_random_samples(self, number_of_samples):
        """Selects a number of samples from the history, chosen at random

        :param number_of_samples: The number of samples to select
        :return: A collection of experiences
        """
        samples = []
        number_of_samples = min(number_of_samples, len(self.history))
        for n in range(number_of_samples):
            samples.append(self.select_random_sample())
        return samples

    def select_random_sample(self):
        """Selects a single random experience from the history

        :return: A single experience
        """
        sample = None
        if len(self.history) > 0:
            index = random.randint(0, len(self.history) - 1)
            sample = copy.copy(self.history[index])
        return sample

    def get_latest_experience(self):
        """Selects the latest experience to be added to the history

        :return: A single experience
        """
        if len(self.history) > 0:
            index = len(self.history) - 1
            sample = self.history[index]
        else:
            sample = None
        return sample

    def select_latest_samples(self, number_of_samples):
        """Selects the newest samples from the history

        :param number_of_samples: The number of samples to select
        :return: A collection of experiences
        """
        if len(self.history) > 0:
            index1 = max(len(self.history) - number_of_samples, 0)
            index2 = len(self.history)
            samples = self.history[index1:index2]
        else:
            samples = [None]
        return samples

    def clear_history(self):
        """Deletes all of the experiences in the history

        :return: None
        """
        self.history = []

    def find_in_history(self, experience):
        """Finds a particular experience in the history, if it exists

        :param experience: The experience to locate
        :return: The index of the experience that is equivalent, or None if there is no equivalent experience
        """
        index = None
        for i in range(len(self.history)):
            if self.history[i].equals(experience):
                index = i
                break
        return index
