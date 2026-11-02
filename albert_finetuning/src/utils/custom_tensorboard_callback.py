from transformers import TrainerCallback
from torch.utils.tensorboard import SummaryWriter
import torch

class CustomTensorBoardCallback(TrainerCallback):
    def __init__(self, tb_writer: SummaryWriter, log_freq: int = 100):
        self.tb_writer = tb_writer
        self.log_freq = log_freq

    def _log_params(self, model, global_step, log_gradients=True):
        for name, param in model.named_parameters():
            # Log only the first layer group's first layer (representative due to weight sharing in ALBERT) and the classifier
            if ("albert.encoder.albert_layer_groups.0.albert_layers.0" in name and ("weight" in name or "bias" in name)) or \
               ("classifier" in name and ("weight" in name or "bias" in name)):
                self.tb_writer.add_histogram(f"weights/{name}", param.data, global_step)
                if log_gradients and param.grad is not None:
                    self.tb_writer.add_histogram(f"gradients/{name}", param.grad, global_step)

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.log_freq == 0 and state.is_world_process_zero:
            model = kwargs['model']
            # Log weights (gradients will be logged in on_after_backward)
            self._log_params(model, state.global_step, log_gradients=False)

    def on_after_backward(self, args, state, control, **kwargs):
        if state.global_step % self.log_freq == 0 and state.is_world_process_zero:
            model = kwargs['model']
            # Log gradients after backward pass, before optimizer step
            self._log_params(model, state.global_step, log_gradients=True)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if state.is_world_process_zero:
            model = kwargs['model']
            # Log weights (and potentially gradients if available) at evaluation steps
            self._log_params(model, state.global_step, log_gradients=True)