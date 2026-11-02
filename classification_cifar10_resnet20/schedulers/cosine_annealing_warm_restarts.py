import torch.optim.lr_scheduler as lr_scheduler
import argparse
import math

def add_cosine_annealing_warm_restarts_args(parser: argparse.ArgumentParser):
    parser.add_argument('--cos_t0', type=int, default=5, help='T_0 for CosineAnnealingWarmRestarts')
    parser.add_argument('--cos_tmult', type=int, default=2, help='T_mult for CosineAnnealingWarmRestarts')
    parser.add_argument('--cos_etamin', type=float, default=0.0, help='eta_min for CosineAnnealingWarmRestarts')
    parser.add_argument('--cos_last_epoch', type=int, default=-1, help='The index of the last epoch for CosineAnnealingWarmRestarts')

def create_cosine_annealing_warm_restarts_scheduler(optimizer, args):
    return lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=args.cos_t0,
        T_mult=args.cos_tmult,
        eta_min=args.cos_etamin,
        last_epoch=args.cos_last_epoch
    )

def append_cosine_annealing_warm_restarts_params(args, name: str):
    name += f'-cosT0{args.cos_t0}'
    if args.cos_tmult != 1:
        name += f'-cosTmult{args.cos_tmult}'
    if args.cos_etamin != 0.0:
        name += f'-cosEtamin{args.cos_etamin}'
    if args.cos_last_epoch != -1:
        name += f'-cosLEpoch{args.cos_last_epoch}'
    return name