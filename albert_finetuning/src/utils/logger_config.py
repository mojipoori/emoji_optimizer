# src/utils/logger_config.py
import logging
import os
from torch.utils.tensorboard import SummaryWriter

def setup_logging(task, log_dir=None):
    """
    Sets up logging and TensorBoard writer with a dynamic log directory.
    
    Args:
        task (str): The task name (e.g., 'cola', 'sts').
        log_dir (str, optional): Path to the TensorBoard log directory. If None, a default is used.
    
    Returns:
        logger: Configured logger object.
        tb_writer: TensorBoard SummaryWriter object.
    """
    # Setup logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Avoid adding multiple handlers if logger is already configured
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Setup TensorBoard writer
    if log_dir is None:
        log_dir = os.path.join('./logs', task)
    
    # Ensure log directory exists
    os.makedirs(log_dir, exist_ok=True)
    
    tb_writer = SummaryWriter(log_dir=log_dir)
    
    return logger, tb_writer