import torch.optim.lr_scheduler as lr_scheduler
import argparse

def add_step_lr_args(parser: argparse.ArgumentParser):
    parser.add_argument('--steplr_stepsize', type=int, default=50, help='Step size for StepLR') # Reverted default to 130 as requested
    parser.add_argument('--steplr_gamma', type=float, default=0.1, help='Gamma for StepLR')
    parser.add_argument('--steplr_last_epoch', type=int, default=-1, help='The index of the last epoch for StepLR')

def create_step_lr_scheduler(optimizer, args):
    return lr_scheduler.StepLR(
        optimizer,
        step_size=args.steplr_stepsize,
        gamma=args.steplr_gamma,
        last_epoch=args.steplr_last_epoch
    )

def append_step_lr_params(args, name: str):
    if args.steplr_stepsize != 10:
        name += f'-slrSS{args.steplr_stepsize}'
    if args.steplr_gamma != 0.5:
        name += f'-slrGam{args.steplr_gamma}'
    if args.steplr_last_epoch != -1:
        name += f'-slrLEpoch{args.steplr_last_epoch}'
    return name