from abc import ABC, abstractmethod



class Logger(ABC):
    @abstractmethod
    def log_scalar(self, tag, value, step=None):
        """
        Logs a scalar value.

        Inputs:
            - tag (str): The tag associated with the scalar value (e.g., "reward/episode_reward").
            - value (float): The scalar value to log.
            - step (int, optional): The global step at which to log the value. If None, it will automatically increment based on the tag starting from 0.
        """
        pass

    def log_scalars(self, tag_value_dict:dict, steps:dict=None):
        """
        Logs multiple scalar values to at once.

        Inputs:
            - tag_value_dict (dict): A dictionary where keys are tags (str) and values are the corresponding scalar values (float) to log.
            - steps (dict, optional): A dictionary where keys are tags (str) and values are the global steps (int) at which to log the corresponding scalar values.
                If None, it will automatically increment based on the tags starting from 0.
        """
        pass

    @abstractmethod
    def close(self):
        pass


class Dummy_Logger(Logger):
    def __init__(self):
        pass

    def log_scalar(self, tag, value, step=None):
        pass

    def log_scalars(self, tag_value_dict:dict, steps:dict=None):
        pass

    def close(self):
        pass