import csv
import os

class SimpleDenomCSV:
    """
    Logs the average denominator actually used in the optimizer step.
    Assumes the optimizer has a method get_mean_denom_and_reset().
    If not, it silently does nothing.
    """
    def __init__(self, log_dir, filename='denom_used.csv'):
        os.makedirs(log_dir, exist_ok=True)
        self.filepath = os.path.join(log_dir, filename)
        self._header_written = False

    def log(self, optimizer, epoch):
        if not hasattr(optimizer, 'get_mean_denom_and_reset'):
            return
        mean_denom = optimizer.get_mean_denom_and_reset()
        if not self._header_written:
            with open(self.filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['epoch', 'mean_denominator'])
            self._header_written = True
        with open(self.filepath, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, mean_denom])