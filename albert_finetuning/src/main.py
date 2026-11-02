# src/main.py
from src.parser.arg_parser import parse_args
from src.tasks.cola import run_cola
from src.tasks.sts import run_sts
from src.tasks.rte import run_rte
from src.tasks.mrpc import run_mrpc
from src.tasks.wnli import run_wnli
from src.tasks.squad_v1 import run_squad_v1
from src.tasks.mnli import run_mnli
from src.tasks.qqp import run_qqp
from src.utils.logger_config import setup_logging
from src.utils.path_utils import generate_run_name
import os
import csv

TASKS = {
    'cola': run_cola,
    'sts': run_sts,
    'rte': run_rte,
    'mrpc': run_mrpc,
    'wnli': run_wnli,
    'squad_v1': run_squad_v1,
    'mnli': run_mnli,
    'qqp': run_qqp
}

def main():
    config, args = parse_args()
    task = args.task
    
    print(f"[DEBUG in main] args.resume_from_checkpoint: {args.resume_from_checkpoint}")
    print(f"[DEBUG in main] config['seed']: {config['seed']}")
    
    # If resuming from checkpoint, run_name is derived from checkpoint path in task function
    # Otherwise, generate new run_name
    if args.resume_from_checkpoint:
        # The task function (e.g., run_mnli) will extract the run_name from the checkpoint path
        # We don't need to generate it here for logging dir
        run_name = os.path.basename(os.path.dirname(args.resume_from_checkpoint))
    else:
        run_name = generate_run_name(config)
    
    print(f"[DEBUG in main] Determined run_name: {run_name}")
    
    log_dir = os.path.join('./logs', run_name)
    logger, tb_writer = setup_logging(task, log_dir=log_dir)
    
    try:
        if task not in TASKS:
            logger.error(f"Task {task} not supported. Choose from {list(TASKS.keys())}")
            raise ValueError(f"Task {task} not supported. Choose from {list(TASKS.keys())}")
        
        trainer = TASKS[task](config, tb_writer, 
                               resume_from_checkpoint=args.resume_from_checkpoint,
                               new_max_steps=args.new_max_steps)
        # trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        # trainer.save_model(os.path.join('./models', run_name))
        logger.info(f"Training completed for {config['task']}")

        # After training, if CSV logging was enabled, read the CSV and write to TensorBoard
        if config.get('optimizer') == 'emoji' and config.get('emoji_log_jerk_to_csv', False):
            csv_file_path = os.path.join(log_dir, "emoji_jerk_log.csv")
            if os.path.exists(csv_file_path):
                logger.info(f"Reading jerk data from {csv_file_path} and writing to TensorBoard.")
                with open(csv_file_path, 'r', newline='') as csvfile:
                    reader = csv.reader(csvfile)
                    header = next(reader) # Skip header row
                    for row in reader:
                        try:
                            step = int(row[0])
                            jerk_before_eps_add = float(row[1])
                            jerk_after_eps_add = float(row[2])
                            tb_writer.add_scalar("jerk_update_term/before_eps_add", jerk_before_eps_add, step)
                            tb_writer.add_scalar("jerk_update_term/after_eps_add", jerk_after_eps_add, step)
                        except (ValueError, IndexError) as e:
                            logger.warning(f"Skipping malformed row in CSV: {row} - Error: {e}")
                logger.info(f"Finished writing jerk data from CSV to TensorBoard.")

        return trainer
    finally:
        tb_writer.close()  # Ensure TensorBoard writer is closed

if __name__ == "__main__":
    main()