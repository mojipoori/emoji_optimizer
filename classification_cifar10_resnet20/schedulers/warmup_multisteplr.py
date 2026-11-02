import argparse
from typing import Dict, Any
import torch.optim.lr_scheduler as lr_scheduler
from .linear_lr import create_linear_lr_scheduler, append_linear_lr_params


def add_warmup_multisteplr_args(parser: argparse.ArgumentParser):
    # Do NOT re-register LinearLR args here (already added in main.py)
    # Only add MultiStepLR args to avoid argparse conflicts
    parser.add_argument('--mslr_milestones', type=int, nargs='+', default=[140, 180], help='Milestones (epochs) for MultiStepLR')
    parser.add_argument('--mslr_gamma', type=float, default=0.1, help='Gamma for MultiStepLR')


def create_warmup_multisteplr_schedulers(optimizer, args) -> Dict[str, Any]:
    warmup = create_linear_lr_scheduler(optimizer, args)
    # Shift MultiStepLR timing one epoch earlier (epoch-based).
    milestones = [int(m) - 1 for m in (args.mslr_milestones or [])]
    main = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=args.mslr_gamma)
    return { 'warmup': warmup, 'main': main }


def append_warmup_multisteplr_params(args, name: str):
    name = append_linear_lr_params(args, name)
    # Append milestones compactly
    if args.mslr_milestones:
        ms = '-'.join(str(m) for m in args.mslr_milestones)
        name += f'-mslrMS{ms}'
    if args.mslr_gamma != 0.1:
        name += f'-mslrGam{args.mslr_gamma}'
    return name
