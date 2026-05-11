from .base_logger import Logger
from pathlib import Path
import uuid
from collections import defaultdict



class TB_Logger(Logger):
    def __init__(self, log_dir):
        """
        A simple wrapper around TensorBoard's SummaryWriter to handle logging and log directory management.

        Inputs:
            - log_dir (str): The directory where TensorBoard logs will be saved.
                If the directory already exists, a unique suffix will be added to avoid overwriting previous logs.
        """
        # Import lazily so TensorBoard is only required when this logger is used.
        # This avoids import errors in environments where tensorboard is not installed.
        from torch.utils.tensorboard import SummaryWriter

        # check if log_dir already exists, if so add a uuid suffix to it to avoid overwriting previous logs
        log_dir_path = Path(log_dir)
        if log_dir_path.exists():
            log_dir = f"{log_dir}_{uuid.uuid4()}"
            print(f"The specified log directory already exists. Using new log directory: {log_dir}")
        else:
            print(f"Using log directory: {log_dir}")

        self.writer = SummaryWriter(log_dir)

        self.global_step_tracker_dict = defaultdict(lambda : 0)  # Dictionary to track global steps for different tags


    def log_scalar(self, tag, value, step=None):
        """
        Logs a scalar value to TensorBoard.

        Inputs:
            - tag (str): The tag associated with the scalar value (e.g., "reward/episode_reward").
            - value (float): The scalar value to log.
            - step (int, optional): The global step at which to log the value. If None, it will automatically increment based on the tag starting from 0.
        """
        if step is None:
            step = self.global_step_tracker_dict[tag]
            self.global_step_tracker_dict[tag] += 1  # Increment the global step for this tag
        self.writer.add_scalar(tag, value, step)


    def log_scalars(self, tag_value_dict:dict, steps:dict=None):
        """
        Logs multiple scalar values to TensorBoard at once.

        Inputs:
            - tag_value_dict (dict): A dictionary where keys are tags (str) and values are the corresponding scalar values (float) to log.
            - steps (dict, optional): A dictionary where keys are tags (str) and values are the global steps (int) at which to log the corresponding scalar values.
                If None, it will automatically increment based on the tags starting from 0.
        """
        for tag, value in tag_value_dict.items():
            # figure out the step to log at for this tag:
            if (steps is not None) and (tag in steps):
                step = steps[tag]
            else:
                step = self.global_step_tracker_dict[tag]
                self.global_step_tracker_dict[tag] += 1  # Increment the global step for this tag

            # log the scalar value for this tag and step
            self.writer.add_scalar(tag, value, step)


    def close(self):
        self.writer.close()