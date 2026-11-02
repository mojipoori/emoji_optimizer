import argparse
from typing import Dict, Any
import torch.optim.lr_scheduler as lr_scheduler
from .linear_lr import create_linear_lr_scheduler, append_linear_lr_params
from .step_lr import append_step_lr_params


def add_warmup_steplr_args(parser: argparse.ArgumentParser):
    # No-op: main parser already registers LinearLR and StepLR args.
    return


def create_warmup_steplr_schedulers(optimizer, args) -> Dict[str, Any]:
    warmup = create_linear_lr_scheduler(optimizer, args)
    # Shift StepLR timing one epoch earlier (epoch-based).
    step_size = max(1, int(args.steplr_stepsize) - 1)
    main = lr_scheduler.StepLR(
        optimizer,
        step_size=step_size,
        gamma=args.steplr_gamma,
        last_epoch=args.steplr_last_epoch,
    )
    return { 'warmup': warmup, 'main': main }


def append_warmup_steplr_params(args, name: str):
    name = append_linear_lr_params(args, name)
    name = append_step_lr_params(args, name)
    return name