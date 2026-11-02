import torch.optim.lr_scheduler as lr_scheduler
import argparse

def add_cyclic_lr_args(parser: argparse.ArgumentParser):
    parser.add_argument('--clr_baselr', type=float, default=0.001, help='Base LR for CyclicLR')
    parser.add_argument('--clr_maxlr', type=float, default=0.006, help='Max LR for CyclicLR')
    parser.add_argument('--clr_stepsize', type=int, default=2000, help='Step size for CyclicLR')
    parser.add_argument('--clr_mode', type=str, default='triangular', choices=['triangular', 'triangular2', 'exp_range'], help='Mode for CyclicLR')
    parser.add_argument('--clr_gamma', type=float, default=1.0, help='Gamma for CyclicLR (exponential range only)')
    parser.add_argument('--clr_scale_fn', type=str, default=None, help='Scale function for CyclicLR')
    parser.add_argument('--clr_scale_mode', type=str, default='cycle', choices=['cycle', 'iterations'], help='Scale mode for CyclicLR')
    parser.add_argument('--clr_cyc_momentum', type=bool, default=True, help='Enable cyclic momentum for CyclicLR')
    parser.add_argument('--clr_basemom', type=float, default=0.8, help='Base momentum for CyclicLR')
    parser.add_argument('--clr_maxmom', type=float, default=0.9, help='Max momentum for CyclicLR')
    parser.add_argument('--clr_last_epoch', type=int, default=-1, help='The index of the last batch for CyclicLR')

def create_cyclic_lr_scheduler(optimizer, args):
    return lr_scheduler.CyclicLR(
        optimizer,
        base_lr=args.clr_baselr,
        max_lr=args.clr_maxlr,
        step_size_up=args.clr_stepsize,
        mode=args.clr_mode,
        gamma=args.clr_gamma,
        scale_fn=None, # args.clr_scale_fn, # Need to handle custom functions if user wants
        scale_mode=args.clr_scale_mode,
        cycle_momentum=args.clr_cyc_momentum,
        base_momentum=args.clr_basemom,
        max_momentum=args.clr_maxmom,
        last_epoch=args.clr_last_epoch
    )

def append_cyclic_lr_params(args, name: str):
    name += f'-clrBLR{args.clr_baselr}'
    name += f'-clrMLR{args.clr_maxlr}'
    if args.clr_stepsize != 2000:
        name += f'-clrSS{args.clr_stepsize}'
    if args.clr_mode != 'triangular':
        name += f'-clrMode{args.clr_mode}'
    if args.clr_gamma != 1.0:
        name += f'-clrGam{args.clr_gamma}'
    if args.clr_scale_mode != 'cycle':
        name += f'-clrSMode{args.clr_scale_mode}'
    if args.clr_cyc_momentum != True:
        name += f'-clrCycMom{args.clr_cyc_momentum}'
    if args.clr_basemom != 0.8:
        name += f'-clrBMom{args.clr_basemom}'
    if args.clr_maxmom != 0.9:
        name += f'-clrMMom{args.clr_maxmom}'
    if args.clr_last_epoch != -1:
        name += f'-clrLEpoch{args.clr_last_epoch}'
    return name