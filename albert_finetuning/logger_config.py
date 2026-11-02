import logging
import os
from torch.utils.tensorboard import SummaryWriter

def setup_logging(task_name, log_dir=None):
    # Create a logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Determine the logs directory
    logs_dir = log_dir if log_dir else os.path.join('./logs', task_name)
    os.makedirs(logs_dir, exist_ok=True)

    # Create a file handler
    log_file = os.path.join(logs_dir, 'training.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create a formatter and set it for both handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Setup TensorBoard writer
    tensorboard_writer = SummaryWriter(log_dir=logs_dir)

    logger.info(f"Logging set up for task: {task_name}")
    return logger, tensorboard_writer