from __future__ import print_function

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

import os
import argparse
import time
import re
from models import *
from optimizers.AdaBelief import AdaBelief
from torch.optim import Adam, SGD, AdamW
from optimizers.Yogi import Yogi
from optimizers.RAdam import RAdam
from optimizers.emoji import Emoji
from optimizers.emojiabl import EmojiAbl
from optimizers.adabound import AdaBound
from optimizers.Lion import Lion
from optimizers.adaboundW import AdaBoundW
from optimizers.SGDW import SGDW
import random
import numpy as np
from denom_logger import SimpleDenomCSV
import math

from schedulers.cosine_annealing_warm_restarts import add_cosine_annealing_warm_restarts_args, create_cosine_annealing_warm_restarts_scheduler, append_cosine_annealing_warm_restarts_params
from schedulers.cyclic_lr import add_cyclic_lr_args, create_cyclic_lr_scheduler, append_cyclic_lr_params
from schedulers.step_lr import add_step_lr_args, create_step_lr_scheduler, append_step_lr_params
from schedulers.linear_lr import add_linear_lr_args, create_linear_lr_scheduler, append_linear_lr_params
from schedulers.warmup_steplr import add_warmup_steplr_args, create_warmup_steplr_schedulers, append_warmup_steplr_params
from schedulers.warmup_multisteplr import add_warmup_multisteplr_args, create_warmup_multisteplr_schedulers, append_warmup_multisteplr_params


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--total_epoch', default=180, type=int, help='Total number of training epochs')
    # parser.add_argument('--decay_epoch', default=None, type=int,
    #                     help='[DEPRECATED] Use --decay_epochs_list instead. Number of epochs to decay learning rate (single epoch).')
    parser.add_argument('--model', default='resnet', type=str, help='model',
                        choices=['resnet', 'resnet18', 'resnet34', 'densenet', 'vgg', 'inception', 'gcvit', 'convformer', 'levit', 'mobilevit', 'regnetx', 'resnet20'])
    parser.add_argument('--optim', default='sgd', type=str, help='optimizer',
                        choices=['sgd', 'sgdw', 'adam', 'adamw', 'adabelief', 'yogi', 'radam', 'adabound', 'lion', 'adaboundw', 'emoji', 'emojiabl'])
    parser.add_argument('--scheduler', default=None, type=str, help='Learning rate scheduler',
                        choices=[None, 'cosineannealingwarmrestarts', 'cycliclr', 'steplr', 'linearlr', 'warmupsteplr', 'warmupmultisteplr'])
    parser.add_argument('--run', default=0, type=int, help='number of runs')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='learning rate decay factor')
    parser.add_argument('--final_lr', default=0.1, type=float,
                        help='final learning rate of AdaBound')
    parser.add_argument('--gamma', default=0.001, type=float,
                        help='convergence speed term of AdaBound')
    parser.add_argument('--eps', default=1e-8, type=float, help='eps for var adam')
    #parser.add_argument('--eps_b', default=1e-8, type=float)

    parser.add_argument('--momentum', default=0.9, type=float, help='momentum term')
    parser.add_argument('--beta1', default=0.9, type=float, help='Adam coefficients beta_1')
    parser.add_argument('--beta2', default=0.999, type=float, help='Adam coefficients beta_2')
    parser.add_argument('--resume', action='store_true', help='resume from checkpoint')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='specific checkpoint file path to resume from')
    parser.add_argument('--batchsize', type=int, default=128, help='batch size')
    parser.add_argument('--weight_decay', default=0, type=float,
                        help='weight decay for optimizers')
    parser.add_argument('--reset', action='store_true',
                        help='whether reset optimizer at learning rate decay')
    parser.add_argument('--rectify', action='store_true', default=False,
                        help='whether to use rectified update for AdaBelief')
    parser.add_argument('--fixed_decay', action='store_true', default=False,
                        help='whether to use fixed weight decay for AdaBelief')
    parser.add_argument('--amsgrad', action='store_true', default=False,
                        help='use AMSGrad for AdaBelief')
    parser.add_argument('--seed', default=1993, type=int, help='Random seed')
    parser.add_argument('--save_interval', default=135, type=int, help='Epoch interval for saving checkpoints. Set to -1 to save only at final epoch, 0 to save only best checkpoint, >0 for regular intervals.')
    parser.add_argument('--decay_checkpoint_folder', type=str, default='decay_checkpoints', help='Subfolder within checkpoint/ to save checkpoints at decay epochs')
    parser.add_argument('--timestamp', action='store_true', help='Add timestamp to checkpoint and log file names')
    parser.add_argument('--histo', action='store_true', default=False, help='Log per-iteration absolute gradient histograms to TensorBoard')
    parser.add_argument('--histo_iter', type=int, default=50, help='Total number of batches to log histograms over entire training when --histo is enabled')
    parser.add_argument('--log_denom', type=bool, default=True,
                    help='Log denominator magnitude (sqrt of second moment) to CSV')
    # Adam-specific arguments
    parser.add_argument('--adam_betas', type=float, nargs=2, default=[0.9, 0.999],
                        help='Adam betas (beta1, beta2)')
    parser.add_argument('--adam_eps', type=float, default=1e-8,
                        help='Adam epsilon for numerical stability')
    parser.add_argument('--adam_amsgrad', type=bool, default=False,
                        help='Use AMSGrad variant for Adam')
    parser.add_argument('--adam_foreach', type=bool, default=False,
                        help='Use foreach implementation for Adam')
    parser.add_argument('--adam_maximize', type=bool, default=False,
                        help='Maximize objective for Adam')
    parser.add_argument('--adam_capturable', type=bool, default=False,
                        help='Use CUDA-capturable state for Adam')
    parser.add_argument('--adam_differentiable', type=bool, default=False,
                        help='Use differentiable optimization for Adam')
    parser.add_argument('--adam_fused', type=bool, default=False,
                        help='Use fused implementation for Adam')

    # AdamW-specific arguments
    parser.add_argument('--adamw_betas', type=float, nargs=2, default=[0.9, 0.999],
                        help='AdamW betas (beta1, beta2)')
    parser.add_argument('--adamw_eps', type=float, default=1e-8,
                        help='AdamW epsilon for numerical stability')
    parser.add_argument('--adamw_amsgrad', type=bool, default=False,
                        help='Use AMSGrad variant for AdamW')
    
    
       

    # SGD-specific arguments
    parser.add_argument('--sgd_momentum', type=float, default=0.9,
                        help='SGD momentum factor')
    parser.add_argument('--sgd_dampening', type=float, default=0.0,
                        help='SGD dampening for momentum')
    parser.add_argument('--sgd_nesterov', type=bool, default=False,
                        help='Use Nesterov momentum for SGD')
    parser.add_argument('--sgdw_weight_decay', type=float, default=5e-4,
                        help='Decoupled weight decay for SGDW')

    # AdaBelief-specific arguments
    parser.add_argument('--adabelief_betas', type=float, nargs=2, default=[0.9, 0.999],
                        help='AdaBelief betas (beta1, beta2)')
    parser.add_argument('--adabelief_eps', type=float, default=1e-8,
                        help='AdaBelief epsilon for numerical stability')
    parser.add_argument('--adabelief_amsgrad', type=bool, default=False,
                        help='Use AMSGrad variant for AdaBelief')
    parser.add_argument('--adabelief_weight_decouple', type=bool, default=True,
                        help='Use decoupled weight decay for AdaBelief')
    parser.add_argument('--adabelief_fixed_decay', type=bool, default=False,
                        help='Use fixed weight decay for AdaBelief')
    parser.add_argument('--adabelief_rectify', type=bool, default=False,
                        help='Use rectified update for AdaBelief')
    parser.add_argument('--adabelief_degenerated_to_sgd', type=bool, default=False,
                        help='Degenerate to SGD for high variance in AdaBelief')

    # AdaBound-specific arguments
    parser.add_argument('--adabound_betas', type=float, nargs=2, default=[0.9, 0.999],
                        help='AdaBound betas (beta1, beta2)')
    parser.add_argument('--adabound_eps', type=float, default=1e-8,
                        help='AdaBound epsilon for numerical stability')
    parser.add_argument('--adabound_final_lr', type=float, default=0.1,
                        help='AdaBound final learning rate')
    parser.add_argument('--adabound_gamma', type=float, default=1e-3,
                        help='AdaBound convergence speed of the bound functions')
    parser.add_argument('--adabound_weight_decay', type=float, default=0.0,
                        help='AdaBound weight decay (L2 penalty)')
    parser.add_argument('--adabound_amsbound', type=bool, default=False,
                        help='Use AMSBound variant for AdaBound')

    # AdaBoundW-specific arguments
    parser.add_argument('--adaboundw_betas', type=float, nargs=2, default=[0.9, 0.999],
                        help='AdaBoundW betas (beta1, beta2)')
    parser.add_argument('--adaboundw_eps', type=float, default=1e-8,
                        help='AdaBoundW epsilon for numerical stability')
    parser.add_argument('--adaboundw_final_lr', type=float, default=0.1,
                        help='AdaBoundW final learning rate')
    parser.add_argument('--adaboundw_gamma', type=float, default=1e-3,
                        help='AdaBoundW convergence speed of the bound functions')
    parser.add_argument('--adaboundw_weight_decay', type=float, default=0.0,
                        help='AdaBoundW weight decay (L2 penalty)')
    parser.add_argument('--adaboundw_amsbound', type=bool, default=False,
                        help='Use AMSBound variant for AdaBoundW')

    # Lion-specific arguments
    parser.add_argument('--lion_lr', type=float, default=1e-4, help='Lion learning rate')
    parser.add_argument('--lion_betas', type=float, nargs=2, default=[0.9, 0.99], help='Lion betas (beta1, beta2)')
    parser.add_argument('--lion_weight_decay', type=float, default=0.0, help='Lion weight decay')


    # emoji-specific arguments
    parser.add_argument('--emoji_betas', type=float, nargs=2, default=[0.9, 0.999],
                        help='emoji betas (beta1, beta2)')
    parser.add_argument('--emoji_eps', type=float, default=1e-8,
                        help='emoji epsilon for numerical stability')
    parser.add_argument('--emoji_weight_decouple', type=bool, default=False,
                        help='Use decoupled weight decay for emoji')
    parser.add_argument(
        '--emoji_amsgrad',
        action='store_true',
        default=False,
        help='Use AMSGrad variant for Emoji',
    )
        
    

    # Yogi-specific arguments
    parser.add_argument('--yogi_lr', type=float, default=1e-2, help='Yogi learning rate')
    parser.add_argument('--yogi_betas', type=float, nargs=2, default=[0.9, 0.999],
                        help='Yogi betas (beta1, beta2)')
    parser.add_argument('--yogi_eps', type=float, default=1e-3,
                        help='Yogi epsilon for numerical stability')
    parser.add_argument('--yogi_initial_accumulator', type=float, default=1e-6,
                        help='Yogi initial accumulator value')
    parser.add_argument('--yogi_weight_decay', type=float, default=0.0,
                        help='Yogi weight decay (L2 penalty)')
    parser.add_argument('--yogi_weight_decouple', type=bool, default=False,
                        help='Use decoupled weight decay for Yogi')

    # RAdam-specific arguments
    parser.add_argument('--radam_betas', type=float, nargs=2, default=[0.9, 0.999],
                        help='RAdam betas (beta1, beta2)')
    parser.add_argument('--radam_eps', type=float, default=1e-8,
                        help='RAdam epsilon for numerical stability')
    parser.add_argument('--radam_weight_decay', type=float, default=0.0,
                        help='RAdam weight decay (L2 penalty)')
    parser.add_argument('--radam_degenerated_to_sgd', type=bool, default=True,
                        help='RAdam degenerated to SGD for high variance')
    parser.add_argument('--alphaa', type=float, default=0.9,
                    help='alphaa for emojig')
    parser.add_argument('--k', type=float, default=10,
                    help='k for emojim')
    parser.add_argument('--adam_w', type=float, default=0.5)
    parser.add_argument('--sgd_w', type=float, default=0.5)
    parser.add_argument('--weight_decouple', type=bool, default=True)

    
    # Add scheduler-specific arguments
    add_cosine_annealing_warm_restarts_args(parser)
    add_cyclic_lr_args(parser)
    add_step_lr_args(parser)
    add_linear_lr_args(parser)
    add_warmup_steplr_args(parser)
    add_warmup_multisteplr_args(parser)

    return parser


def build_dataset(args):
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                            transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batchsize, shuffle=True,
                                               num_workers=0)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                           transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batchsize, shuffle=False, num_workers=0)

    return train_loader, test_loader


def get_ckpt_name(args):
    name_parts = []
    
    # Add model and optimizer
    name_parts.append(f'{args.model}-{args.optim}')
    
    # Add optimizer-specific parameters
    optim_specific_part = {
        'sgd': 'lr{}-mom{}-wdecay{}-run{}-seed{}'.format(args.lr, args.momentum, args.weight_decay, args.run, args.seed) +
               ('-damp{}'.format(args.sgd_dampening) if args.sgd_dampening != 0.0 else '') +
               ('-nesterov' if args.sgd_nesterov != False else ''),
        'sgdw': 'lr{}-mom{}-wd{}-run{}-seed{}'.format(args.lr, args.sgd_momentum, args.sgdw_weight_decay, args.run, args.seed) +
                ('-damp{}'.format(args.sgd_dampening) if args.sgd_dampening != 0.0 else '') +
                ('-nesterov' if args.sgd_nesterov != False else ''),
        'adam': 'lr{}-betas{}-{}-wdecay{}-eps{}-run{}-seed{}'.format(args.lr, args.adam_betas[0], args.adam_betas[1], args.weight_decay, args.adam_eps, args.run, args.seed) +
                ('-amsg' if args.adam_amsgrad != False else '') +
                ('-foreach' if args.adam_foreach != False else '') +
                ('-max' if args.adam_maximize != False else '') +
                ('-cap' if args.adam_capturable != False else '') +
                ('-diff' if args.adam_differentiable != False else '') +
                ('-fused' if args.adam_fused != False else ''),

        'radam': 'lr{}-betas{}-{}-wdecay{}-eps{}-run{}-seed{}'.format(args.lr, args.radam_betas[0], args.radam_betas[1], args.radam_weight_decay, args.radam_eps, args.run, args.seed) +
                 ('-degSGD' if args.radam_degenerated_to_sgd != True else ''),
        'adamw': 'lr{}-betas{}-{}-wdecay{}-eps{}-run{}-seed{}'.format(args.lr, args.adamw_betas[0], args.adamw_betas[1], args.weight_decay, args.adamw_eps, args.run, args.seed) +
                 ('-amsg' if args.adamw_amsgrad != False else ''),
                
        'adabelief': 'lr{}-betas{}-{}-eps{}-wdecay{}-rectify{}-fixed_decay{}-amsgrad{}-run{}-seed{}'.format(
            args.lr, args.adabelief_betas[0], args.adabelief_betas[1], args.eps, args.weight_decay, args.adabelief_rectify, args.adabelief_fixed_decay, args.adabelief_amsgrad, args.run, args.seed),
        'adabound': 'lr{}-betas{}-{}-final_lr{}-gamma{}-wdecay{}-amsbound{}-run{}-seed{}'.format(
            args.lr, args.adabound_betas[0], args.adabound_betas[1], args.adabound_final_lr, args.adabound_gamma, args.adabound_weight_decay, args.adabound_amsbound, args.run, args.seed),
        'yogi': 'lr{}-betas{}-{}-eps{}-wdecay{}-run{}-seed{}'.format(args.yogi_lr, args.yogi_betas[0], args.yogi_betas[1], args.yogi_eps, args.yogi_weight_decay, args.run, args.seed) +
                ('-initacc{}'.format(args.yogi_initial_accumulator) if args.yogi_initial_accumulator != 1e-6 else '') +
                ('-wdcpl' if args.yogi_weight_decouple != False else ''),
        'emoji': 'lr{}-betas{}-{}-eps{}-wdecay{}-run{}-seed{}'.format(
            args.lr,
            args.emoji_betas[0],
            args.emoji_betas[1],
            args.emoji_eps,
            args.weight_decay,
            args.run,
            args.seed,
        ) + \
            ('-wdcpl' if args.emoji_weight_decouple else '') + \
            ('-amsg' if args.emoji_amsgrad else ''),
       
        'emojiabl': 'lr{}-betas{}-{}-eps{}-wdecay{}-run{}-seed{}-alphaa{}'.format(
            args.lr,
            args.emoji_betas[0],
            args.emoji_betas[1],
            args.emoji_eps,
            args.weight_decay,
            args.run,
            args.seed,
            args.alphaa,
        ) + \
            ('-wdcpl' if args.emoji_weight_decouple else '') + \
            ('-amsg' if args.emoji_amsgrad else ''),        
        
        'adaboundw': 'lr{}-betas{}-{}-final_lr{}-gamma{}-wdecay{}-eps{}-amsbound{}-run{}-seed{}'.format(
            args.lr, args.adaboundw_betas[0], args.adaboundw_betas[1], args.adaboundw_final_lr, args.adaboundw_gamma, args.adaboundw_weight_decay, args.adaboundw_eps, args.adaboundw_amsbound, args.run, args.seed),
        'lion': 'lr{}-betas{}-{}-wdecay{}-run{}-seed{}'.format(args.lion_lr, args.lion_betas[0], args.lion_betas[1], args.lion_weight_decay, args.run, args.seed),
    }[args.optim]
    name_parts.append(optim_specific_part)

    # Add scheduler parameters to the checkpoint name if a scheduler is chosen
    if args.scheduler is not None:
        name_parts.append(args.scheduler)
        if args.scheduler == 'cosineannealingwarmrestarts':
            name_parts.append(append_cosine_annealing_warm_restarts_params(args, ''))
        elif args.scheduler == 'cycliclr':
            name_parts.append(append_cyclic_lr_params(args, ''))
        elif args.scheduler == 'steplr':
            name_parts.append(append_step_lr_params(args, ''))
        elif args.scheduler == 'linearlr':
            name_parts.append(append_linear_lr_params(args, ''))
        elif args.scheduler == 'warmupsteplr':
            name_parts.append(append_warmup_steplr_params(args, ''))
        elif args.scheduler == 'warmupmultisteplr':
            name_parts.append(append_warmup_multisteplr_params(args, ''))

    # Add general arguments if they are not at their default values
    if args.batchsize != 128:
        name_parts.append(f'-bs{args.batchsize}')
    if args.momentum != 0.9 and args.optim == 'sgd': # Only add momentum if SGD and not default
        name_parts.append(f'-mom{args.momentum}')
    if args.weight_decay != 0 and args.optim not in ['adam', 'adamw', 'adabelief', 'yogi', 'msvag', 'radam', 'adabound', 'emoji', 'miadam1']:
        name_parts.append(f'-wd{args.weight_decay}')
    if args.eps != 1e-8 and args.optim not in ['adam', 'adamw', 'adabelief', 'yogi', 'msvag', 'radam', 'adabound', 'emoji', 'fromage', 'miadam1']:
        name_parts.append(f'-eps{args.eps}')

    # Add timestamp if requested
    if args.timestamp:
        name_parts.append(time.strftime("%Y%m%d-%H%M%S"))

    # Join all parts and add reset status
    return '{}-reset{}'.format('-'.join(filter(None, name_parts)), str(args.reset))


def load_checkpoint(ckpt_name):
    print('==> Resuming from checkpoint..')
    path = os.path.join('checkpoint', ckpt_name)
    if not os.path.exists(path):
        print('Error: checkpoint {} not found'.format(path))
        return None
    return torch.load(path)


def build_model(args, device, ckpt=None):
    print('==> Building model..')
    net = {
        'resnet': ResNet34,
        'resnet18': ResNet18,
        'resnet34': ResNet34,
        'resnet20': lambda: resnet(depth=20, num_classes=10, block_name='BasicBlock'),
        'densenet': DenseNet121,
        'vgg': vgg11,
        'inception': inception_next_tiny,
        'gcvit': gcvit_xxtiny,
        'convformer': convformer_s18,
        'levit': levit_128s,
        'mobilevit': mobilevit_s,
        'regnetx': regnetx_016,
    }[args.model]()
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if ckpt:
        net.load_state_dict(ckpt['net'])

    return net


def get_optimizer_lr(args):
    """
    Get the appropriate learning rate parameter for the specified optimizer.
    Each optimizer has its own LR parameter name.
    
    Returns:
        float: The learning rate value for the optimizer
    """
    args.optim = args.optim.lower()
    
    # Map each optimizer to its specific LR parameter
    lr_mapping = {
        'sgd': args.lr,
        'sgdw': args.lr,
        'adam': args.lr,
        'adamw': args.lr,
        'adabelief': args.lr,
        'radam': args.lr,
        'emoji': args.lr,
        'adabound': args.lr,
        'adaboundw': args.lr,
        'yogi': args.yogi_lr,
        'lion': args.lion_lr,
        'emojiabl': args.lr,
    }
    
    if args.optim in lr_mapping:
        lr_value = lr_mapping[args.optim]
        print(f"DEBUG: Using LR parameter for {args.optim}: {lr_value}")
        return lr_value
    else:
        print(f"WARNING: Unknown optimizer {args.optim}, using default --lr: {args.lr}")
        return args.lr


def create_optimizer(args, model_params):
    args.optim = args.optim.lower()
    
    # Get the appropriate LR for this optimizer
    optimizer_lr = get_optimizer_lr(args)
    
    if args.optim == 'sgd':
        optimizer = optim.SGD(model_params, optimizer_lr, momentum=args.sgd_momentum,
                                     weight_decay=args.weight_decay, dampening=args.sgd_dampening, nesterov=args.sgd_nesterov)
    elif args.optim == 'sgdw':
        optimizer = SGDW(model_params, lr=optimizer_lr, momentum=args.sgd_momentum,
                    weight_decay=args.sgdw_weight_decay, dampening=args.sgd_dampening, nesterov=args.sgd_nesterov)
    elif args.optim == 'adam':
        optimizer = Adam(model_params, lr=optimizer_lr, betas=args.adam_betas,
                    weight_decay=args.weight_decay, eps=args.adam_eps,
                    amsgrad=args.adam_amsgrad, foreach=args.adam_foreach,
                    maximize=args.adam_maximize, capturable=args.adam_capturable,
                    differentiable=args.adam_differentiable, fused=args.adam_fused)
    elif args.optim == 'radam':
        optimizer = RAdam(model_params, optimizer_lr, betas=(args.radam_betas[0], args.radam_betas[1]),
                                 weight_decay=args.radam_weight_decay, eps=args.radam_eps)
    elif args.optim == 'adamw':
        optimizer = AdamW(model_params, lr=optimizer_lr, betas=args.adamw_betas,
                                 weight_decay=args.weight_decay, eps=args.adamw_eps,
                                 amsgrad=args.adamw_amsgrad)       
    elif args.optim == 'adabelief':
        optimizer = AdaBelief(model_params, optimizer_lr, betas=(args.adabelief_betas[0], args.adabelief_betas[1]),
                                     weight_decay=args.weight_decay, eps=args.eps,
                                     rectify=args.adabelief_rectify, fixed_decay=args.adabelief_fixed_decay, amsgrad=args.adabelief_amsgrad,
                                     weight_decouple=args.adabelief_weight_decouple, degenerated_to_sgd=args.adabelief_degenerated_to_sgd)
    elif args.optim == 'yogi':
        optimizer = Yogi(model_params, optimizer_lr, betas=args.yogi_betas,
                            weight_decay=args.yogi_weight_decay, eps=args.yogi_eps,
                            initial_accumulator=args.yogi_initial_accumulator,
                            weight_decouple=args.yogi_weight_decouple)
    elif args.optim == 'emoji':
        optimizer = Emoji(model_params, lr=optimizer_lr, 
                         betas=args.emoji_betas, 
                         eps=args.emoji_eps, 
                         weight_decay=args.weight_decay,
                         weight_decouple=args.emoji_weight_decouple,
                         amsgrad=args.emoji_amsgrad,
                         )


    elif args.optim == 'emojiabl':
        optimizer = EmojiAbl(model_params, lr=optimizer_lr, 
                         betas=args.emoji_betas, 
                         eps=args.emoji_eps, 
                         weight_decay=args.weight_decay,
                         weight_decouple=args.emoji_weight_decouple,
                         amsgrad=args.emoji_amsgrad,
                         alphaa=args.alphaa
                         )        
        
                           
       
    elif args.optim == 'adabound':
        optimizer = AdaBound(model_params, lr=optimizer_lr, betas=args.adabound_betas,
                                    final_lr=args.adabound_final_lr, gamma=args.adabound_gamma,
                                    weight_decay=args.adabound_weight_decay, eps=args.adabound_eps,
                                    amsbound=args.adabound_amsbound)
    elif args.optim == 'adaboundw':
        optimizer = AdaBoundW(model_params, lr=optimizer_lr, betas=args.adaboundw_betas,
                                     final_lr=args.adaboundw_final_lr, gamma=args.adaboundw_gamma,
                                     weight_decay=args.adaboundw_weight_decay, eps=args.adaboundw_eps,
                                     amsbound=args.adaboundw_amsbound)
    elif args.optim == 'lion':
        optimizer = Lion(model_params, lr=optimizer_lr, betas=args.lion_betas,
                            weight_decay=args.lion_weight_decay)
    else:
        print('Optimizer not found')
        return None
    
    # Ensure 'initial_lr' is set in param_groups for scheduler compatibility
    for group in optimizer.param_groups:
        group.setdefault('initial_lr', group['lr'])

    return optimizer


def create_scheduler(optimizer, args):
    scheduler = None
    if args.scheduler == 'cosineannealingwarmrestarts':
        scheduler = create_cosine_annealing_warm_restarts_scheduler(optimizer, args)
    elif args.scheduler == 'cycliclr':
        scheduler = create_cyclic_lr_scheduler(optimizer, args)
    elif args.scheduler == 'steplr':
        scheduler = create_step_lr_scheduler(optimizer, args)
    elif args.scheduler == 'linearlr':
        scheduler = create_linear_lr_scheduler(optimizer, args)
    elif args.scheduler == 'warmupsteplr':
        scheduler = create_warmup_steplr_schedulers(optimizer, args)
    elif args.scheduler == 'warmupmultisteplr':
        scheduler = create_warmup_multisteplr_schedulers(optimizer, args)
    return scheduler


def train(net, epoch, device, data_loader, optimizer, criterion, args, writer, scheduler=None, warmup_recorder=None, ckpt_name=None):
    print('Epoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    with tqdm(total=len(data_loader), desc=f'Training Epoch {epoch}', unit='batch') as pbar:
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            # Record absolute gradient histogram per-iteration only when --histo is enabled
            if getattr(args, 'histo', False) and getattr(args, '_histo_logged', 0) < getattr(args, 'histo_iter', 30):
                try:
                    with torch.no_grad():
                        all_grads = []
                        for param in net.parameters():
                            if param.grad is not None:
                                grad_vals = param.grad.detach().abs().view(-1)
                                if grad_vals.numel() > 0:
                                    all_grads.append(grad_vals)
                        if all_grads:
                            grads_abs = torch.cat(all_grads).cpu()
                            # Log-scale x-axis by storing ln(|g| + eps)
                            log_grads = torch.log(grads_abs + 1e-20)
                            global_step = epoch * len(data_loader) + batch_idx
                            writer.add_histogram('Gradients/abs_ln', log_grads, global_step=global_step)
                            # Increase global count
                            args._histo_logged = getattr(args, '_histo_logged', 0) + 1
                except Exception as e:
                    print(f"DEBUG: Failed to record gradient histogram: {e}")
            optimizer.step()
 
            # Record LR during warmup (per-iteration) if requested
            if warmup_recorder is not None and args.scheduler in ['warmupsteplr', 'warmupmultisteplr']:
                try:
                    current_lr = optimizer.param_groups[0]['lr']
                    # Collect only during warmup window
                    warm = scheduler.get('warmup') if isinstance(scheduler, dict) else None
                    if warm and warm.last_epoch < warm.total_iters:
                        warmup_recorder.setdefault('lrs', []).append(current_lr)
                        warmup_recorder.setdefault('iterations', []).append(warm.last_epoch)
                        warmup_recorder.setdefault('epochs', []).append(epoch)
                        warmup_recorder.setdefault('batch_indices', []).append(batch_idx)
                        
                        # Save to CSV file during warmup
                        if not warmup_recorder.get('csv_file_created', False):
                            import csv
                            csv_filename = f"warmup_lr_data_{ckpt_name}.csv"
                            warmup_recorder['csv_filename'] = csv_filename
                            with open(csv_filename, 'w', newline='') as csvfile:
                                csv_writer = csv.writer(csvfile)
                                csv_writer.writerow(['Epoch', 'Batch', 'Iteration', 'Learning_Rate'])
                            warmup_recorder['csv_file_created'] = True
                            print(f"DEBUG: Created CSV file: {csv_filename}")
                        
                        # Append to CSV file
                        import csv
                        with open(warmup_recorder['csv_filename'], 'a', newline='') as csvfile:
                            csv_writer = csv.writer(csvfile)
                            csv_writer.writerow([epoch, batch_idx, warm.last_epoch, current_lr])
                        
                        # Debug: print warmup recording progress
                        if batch_idx == 0 and epoch == 0:  # Print only once
                            print(f"DEBUG: Recording warmup LR: {current_lr}, total recorded: {len(warmup_recorder.get('lrs', []))}")
                except Exception as e:
                    print(f"DEBUG: Error recording warmup LR: {e}")
                    pass

            if scheduler:
                if args.scheduler in ['cycliclr', 'linearlr']:
                    scheduler.step()
                elif args.scheduler in ['warmupsteplr', 'warmupmultisteplr']:
                    # step warmup per-iteration until it finishes
                    warm = scheduler.get('warmup')
                    if warm and warm.last_epoch < warm.total_iters:
                        warm.step()
                        # Debug: print warmup progress
                        if batch_idx == 0:  # Print only once per epoch
                            print(f"Warmup: last_epoch={warm.last_epoch}, total_iters={warm.total_iters}, current_lr={optimizer.param_groups[0]['lr']:.6f}")
                        # When warmup just finished, save plot once
                        if warm.last_epoch >= warm.total_iters and warmup_recorder is not None and not warmup_recorder.get('done_saved', False):
                            try:
                                import os
                                import matplotlib.pyplot as plt
                                import csv
                                os.makedirs('lr_plots', exist_ok=True)
                                
                                lrs = warmup_recorder.get('lrs', [])
                                iterations = warmup_recorder.get('iterations', [])
                                epochs = warmup_recorder.get('epochs', [])
                                batch_indices = warmup_recorder.get('batch_indices', [])
                                
                                print(f"DEBUG: Saving warmup plot with {len(lrs)} LR values")
                                
                                # Create detailed plot
                                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
                                
                                # Plot 1: LR vs Iteration
                                ax1.plot(iterations, lrs, 'b-', linewidth=2, label='Warmup LR')
                                ax1.set_xlabel('Iteration')
                                ax1.set_ylabel('Learning Rate')
                                ax1.set_title('Warmup LR Curve')
                                ax1.grid(True, alpha=0.3)
                                ax1.legend()
                                
                                # Add annotations for key points
                                if len(lrs) > 0:
                                    ax1.annotate(f'Start: {lrs[0]:.2e}', 
                                                xy=(iterations[0], lrs[0]), 
                                                xytext=(50, 20), 
                                                textcoords='offset points',
                                                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                                                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
                                    
                                    ax1.annotate(f'End: {lrs[-1]:.2e}', 
                                                xy=(iterations[-1], lrs[-1]), 
                                                xytext=(-50, 20), 
                                                textcoords='offset points',
                                                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7),
                                                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
                                
                                # Plot 2: LR vs Epoch (if multiple epochs)
                                if len(set(epochs)) > 1:
                                    ax2.plot(epochs, lrs, 'r-', linewidth=2, label='Warmup LR by Epoch')
                                    ax2.set_xlabel('Epoch')
                                    ax2.set_ylabel('Learning Rate')
                                    ax2.set_title('Warmup LR by Epoch')
                                    ax2.grid(True, alpha=0.3)
                                    ax2.legend()
                                else:
                                    ax2.text(0.5, 0.5, f'Warmup completed in epoch {epochs[0] if epochs else 0}', 
                                            ha='center', va='center', transform=ax2.transAxes, fontsize=14)
                                    ax2.set_title('Warmup Summary')
                                
                                plt.tight_layout()
                                out_name = f"lr_plots/warmup_{ckpt_name or 'run'}.png"
                                plt.savefig(out_name, dpi=150, bbox_inches='tight')
                                plt.close()
                                
                                # Save summary to text file
                                summary_filename = f"warmup_summary_{ckpt_name}.txt"
                                with open(summary_filename, 'w') as f:
                                    f.write("=== Warmup Scheduler Summary ===\n")
                                    f.write(f"Total iterations: {len(lrs)}\n")
                                    f.write(f"Start LR: {lrs[0] if lrs else 'N/A'}\n")
                                    f.write(f"End LR: {lrs[-1] if lrs else 'N/A'}\n")
                                    f.write(f"Epochs covered: {min(epochs) if epochs else 'N/A'} to {max(epochs) if epochs else 'N/A'}\n")
                                    f.write(f"CSV file: {warmup_recorder.get('csv_filename', 'N/A')}\n")
                                    f.write(f"Plot file: {out_name}\n")
                                
                                print(f"DEBUG: Warmup plot saved to {out_name}")
                                print(f"DEBUG: Summary saved to {summary_filename}")
                                warmup_recorder['done_saved'] = True
                            except Exception as e:
                                print(f"DEBUG: Error saving warmup plot: {e}")
                                pass
 
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
 
            pbar.set_postfix({'loss': train_loss / (batch_idx + 1), 'acc': 100. * correct / total})
            pbar.update(1)

    accuracy = 100. * correct / total
    print('train acc %.3f' % accuracy)
    writer.add_scalar('Loss/train', train_loss / (batch_idx + 1), epoch)
    writer.add_scalar('Accuracy/train', accuracy, epoch)
    
    # Log learning rate to TensorBoard
    current_lr = optimizer.param_groups[0]['lr']
    writer.add_scalar('Learning_Rate', current_lr, epoch)

    return accuracy


def test(net, device, data_loader, criterion, writer, epoch):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with tqdm(total=len(data_loader), desc=f'Testing Epoch {epoch}', unit='batch') as pbar:
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(data_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                pbar.set_postfix({'loss': test_loss / (batch_idx + 1), 'acc': 100. * correct / total})
                pbar.update(1)

    accuracy = 100. * correct / total
    print('test acc %.3f' % accuracy)
    writer.add_scalar('Loss/test', test_loss / (batch_idx + 1), epoch)
    writer.add_scalar('Accuracy/test', accuracy, epoch)
    
    # Log learning rate to TensorBoard (same as in train function)
    # Note: This will be the same LR as logged in train function

    return accuracy


def check_config_compatibility(args, ckpt):
    if not ckpt:
        return False
    ckpt_args = ckpt.get('args', {})
    
    # Check optimizer and scheduler compatibility
    optim_params = ['optim', 'lr', 'final_lr', 'momentum', 'beta1', 'beta2', 'beta3', 'gamma', 'eps', 'weight_decay', 'reset', 'run']
    scheduler_params = ['scheduler', 'cos_t0', 'cos_tmult', 'cos_etamin', 'clr_baselr', 'clr_maxlr', 'clr_stepsize', 'clr_mode', 'clr_gamma', 'clr_scale_mode', 'clr_cyc_momentum', 'clr_basemom', 'clr_maxmom', 'steplr_stepsize', 'steplr_gamma', 'clr_last_epoch'] # Revert scheduler parameters to compatibility check
    
    for param in optim_params:  # Only check optimizer parameters, ignore scheduler
        if ckpt_args.get(param) != vars(args).get(param):
            print(f"Incompatible parameter: {param}, Checkpoint: {ckpt_args.get(param)}, Current: {vars(args).get(param)}")
            return False

    other_params = ['model', 'batchsize', 'seed']
    for param in other_params:
        if ckpt_args.get(param) != vars(args).get(param):
            print(f"Incompatible parameter: {param}, Checkpoint: {ckpt_args.get(param)}, Current: {vars(args).get(param)}")
            return False
    return True

def print_optimizer_memory(optimizer):
    total_bytes = 0
    state_memory = {}

    for state in optimizer.state.values():
        for state_name, value in state.items():
            if not torch.is_tensor(value):
                continue

            bytes_used = value.numel() * value.element_size()
            total_bytes += bytes_used
            state_memory[state_name] = state_memory.get(state_name, 0) + bytes_used

    print("\n" + "=" * 70)
    print("OPTIMIZER STATE MEMORY")
    print("=" * 70)

    for state_name, bytes_used in state_memory.items():
        print(
            f"{str(state_name):<25} "
            f"{bytes_used / (1024 ** 2):>10.2f} MB"
        )

    print("-" * 70)
    print(f"Total optimizer state: {total_bytes / (1024 ** 2):.2f} MB")
    print(f"Total optimizer state: {total_bytes / (1024 ** 3):.4f} GB")
    print("=" * 70)


def print_gpu_memory():
    torch.cuda.synchronize()

    print("\n" + "=" * 70)
    print("GPU MEMORY")
    print("=" * 70)

    print(
        f"Current allocated: "
        f"{torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB"
    )

    print(
        f"Peak allocated:    "
        f"{torch.cuda.max_memory_allocated() / (1024 ** 2):.2f} MB"
    )

    print(
        f"Current reserved:  "
        f"{torch.cuda.memory_reserved() / (1024 ** 2):.2f} MB"
    )

    print(
        f"Peak reserved:     "
        f"{torch.cuda.max_memory_reserved() / (1024 ** 2):.2f} MB"
    )

    print("=" * 70)


def main():
    parser = get_parser()
    args = parser.parse_args()

    print(f"DEBUG: args.steplr_stepsize after parsing: {args.steplr_stepsize}")
    print(f"DEBUG: args.steplr_last_epoch after parsing: {args.steplr_last_epoch}")
    
    set_seed(args.seed)
    # Initialize global histogram counter (total across training)
    setattr(args, '_histo_logged', 0)

    train_loader, test_loader = build_dataset(args)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ckpt_name = get_ckpt_name(args)
    if args.resume:
        if args.checkpoint_path:
            # Use specific checkpoint path if provided
            ckpt_name = args.checkpoint_path
            print(f'Using specific checkpoint path: {ckpt_name}')
        else:
            # Strip trailing timestamp segment in name for resume matching (pattern YYYYMMDD-HHMMSS)
            ckpt_name = re.sub(r"-\d{8}-\d{6}(?=-resetFalse$)", "", ckpt_name)
            # Also strip steplr_stepsize from name for resume matching since it shouldn't block resume
            if args.scheduler in ['steplr', 'warmupsteplr', 'warmupmultisteplr']:
                ckpt_name = re.sub(r"-slrSS\d+", "", ckpt_name)
            print('Modified ckpt_name for resume:', ckpt_name)
    print('ckpt_name:', ckpt_name)

    # Create unique log directory for resume to avoid overwriting previous runs
    if args.resume:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        log_dir = os.path.join('runs', f"{ckpt_name}-resume-{timestamp}")
    else:
        log_dir = os.path.join('runs', ckpt_name)
    writer = SummaryWriter(log_dir=log_dir)
    # ---- Denominator logger ----
    if args.log_denom:
        denom_csv = SimpleDenomCSV(log_dir)
    else:
        denom_csv = None
 
            
    ckpt = load_checkpoint(ckpt_name)
    
    # If resume and no checkpoint found, try to find one with timestamp
    if args.resume and ckpt is None:
        # Look for checkpoints with timestamp that match the base name
        import glob
        # Create a pattern that allows for different steplr_stepsize values and timestamps
        if args.scheduler in ['steplr', 'warmupsteplr', 'warmupmultisteplr']:
            # If current name lacks -slrSS..., inject wildcard before -slrGam; otherwise, generalize existing -slrSS...
            if '-slrSS' not in ckpt_name:
                pattern = ckpt_name.replace('-slrGam', '-slrSS*-slrGam')
            else:
                pattern = re.sub(r'-slrSS\d+', '-slrSS*', ckpt_name)
            pattern = pattern.replace('-resetFalse', '-*-resetFalse')
        else:
            pattern = ckpt_name.replace('-resetFalse', '-*-resetFalse')
        
        matching_files = glob.glob(os.path.join('checkpoint', pattern))
        if matching_files:
            # Use the first matching file (latest by timestamp if multiple)
            matching_files.sort()
            ckpt_name = os.path.basename(matching_files[-1])
            print(f'Found checkpoint with timestamp: {ckpt_name}')
            ckpt = load_checkpoint(ckpt_name)
    start_epoch = -1
    best_acc = 0.0
    train_accuracies = []
    test_accuracies = []
    
    if args.resume and ckpt:
        best_acc = ckpt['acc']
        start_epoch = ckpt['epoch']

        # If resuming with CosineAnnealingWarmRestarts, set last_epoch BEFORE compatibility check
        if args.scheduler == 'cosineannealingwarmrestarts':
            args.cos_last_epoch = start_epoch
        # If resuming with StepLR, set last_epoch BEFORE compatibility check
        elif args.scheduler == 'steplr':
            args.steplr_last_epoch = start_epoch

        # Do not let --total_epoch block continuation when resuming
        if args.total_epoch <= start_epoch:
            args.total_epoch = start_epoch + 1

        if check_config_compatibility(args, ckpt):
            print('Resuming from compatible checkpoint')
            curve = os.path.join('curve', ckpt_name)
            curve_data = torch.load(curve) if os.path.exists(curve) else {'train_acc': [], 'test_acc': []}
            train_accuracies = curve_data['train_acc']
            test_accuracies = curve_data['test_acc']
        else:
            print('Checkpoint found but settings are incompatible. Starting new run.')
            ckpt = None # Ensure we start fresh if incompatible
    # The previous else branch (lines 597-602) for incompatible settings is removed.
    # The idea is that if args.resume is true and a checkpoint exists, we always try to load it.
    # Any potential incompatibilities in scheduler parameters will be handled by loading the scheduler state,
    # which should override the argparse defaults.

    net = build_model(args, device, ckpt=ckpt)
    criterion = torch.nn.CrossEntropyLoss()
    
    optimizer = create_optimizer(args, net.parameters())

    # Load optimizer state if resuming
    if args.resume and ckpt and start_epoch >= 0:
        optimizer.load_state_dict(ckpt['optimizer'])
        # Ensure initial_lr is set for scheduler compatibility
        for param_group in optimizer.param_groups:
            if 'initial_lr' not in param_group:
                param_group['initial_lr'] = param_group['lr']

    print(f"DEBUG: args.steplr_stepsize before scheduler creation: {args.steplr_stepsize}")
    print(f"DEBUG: args.steplr_gamma before scheduler creation: {args.steplr_gamma}")
    print(f"DEBUG: args.steplr_last_epoch before scheduler creation: {args.steplr_last_epoch}")

    # Create scheduler AFTER potentially loading optimizer state
    scheduler = create_scheduler(optimizer, args)

    if isinstance(scheduler, dict):
        warm = scheduler.get('warmup')
        main_sched = scheduler.get('main')
        try:
            print(f"DEBUG: warmup.total_iters: {getattr(warm, 'total_iters', 'N/A')}, warmup.last_epoch: {getattr(warm, 'last_epoch', 'N/A')}")
        except Exception:
            pass
        try:
            print(f"DEBUG: steplr.step_size: {getattr(main_sched, 'step_size', 'N/A')}, steplr.last_epoch: {getattr(main_sched, 'last_epoch', 'N/A')}")
        except Exception:
            pass
    else:
        try:
            print(f"DEBUG: scheduler.last_epoch after creation: {getattr(scheduler, 'last_epoch', 'N/A')}")
        except Exception:
            pass

    # Always ensure scheduler starts from epoch 0 and learning rate is correct
    if scheduler:
        print(f"DEBUG: Ensuring scheduler starts from epoch 0")
        if isinstance(scheduler, dict):
            warm = scheduler.get('warmup')
            main_sched = scheduler.get('main')
            if main_sched:
                # Reset main scheduler to start from epoch 0
                main_sched.last_epoch = -1
                print(f"DEBUG: Reset main scheduler last_epoch to -1")
        else:
            # Reset single scheduler to start from epoch 0
            scheduler.last_epoch = -1
            print(f"DEBUG: Reset scheduler last_epoch to -1")
    
    # Ensure learning rate is set to initial value (but don't override warmup scheduler settings)
    # NOTE: This check happens AFTER scheduler creation, so we can properly detect warmup schedulers
    if not (scheduler and isinstance(scheduler, dict) and scheduler.get('warmup')):
        # Only set LR to optimizer LR if we don't have a warmup scheduler
        optimizer_lr = get_optimizer_lr(args)
        for param_group in optimizer.param_groups:
            param_group['lr'] = optimizer_lr
            print(f"DEBUG: Set learning rate to {optimizer_lr}")
    else:
        print(f"DEBUG: Skipping LR reset because warmup scheduler will handle it")
    
    # Print current LR
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Current Learning Rate: {current_lr}")

    if args.resume and ckpt and start_epoch >= 0:
        # optimizer.load_state_dict(ckpt['optimizer']) # already loaded above
        if scheduler and 'scheduler' in ckpt:
            try:
                if isinstance(scheduler, dict):
                    # For warmup schedulers, load both warmup and main scheduler states
                    scheduler_state = ckpt['scheduler']
                    if 'warmup' in scheduler_state and scheduler_state['warmup'] is not None:
                        scheduler['warmup'].load_state_dict(scheduler_state['warmup'])
                        print(f"DEBUG: Loaded warmup scheduler state - last_epoch: {scheduler['warmup'].last_epoch}, total_iters: {scheduler['warmup'].total_iters}")
                    if 'main' in scheduler_state and scheduler_state['main'] is not None:
                        scheduler['main'].load_state_dict(scheduler_state['main'])
                        print(f"DEBUG: Loaded main scheduler state")
                else:
                    # For regular schedulers
                    scheduler.load_state_dict(ckpt['scheduler'])
                    print(f"DEBUG: Loaded scheduler state")
            except Exception as e:
                print(f"WARNING: Failed to load scheduler state: {e}")
                print(f"DEBUG: Continuing without scheduler state restoration")
        
    state = {
        'args': vars(args),
        'optimizer': optimizer.state_dict(),
    }
    if scheduler:
        try:
            if isinstance(scheduler, dict):
                warm = scheduler.get('warmup')
                main_sched = scheduler.get('main')
                state['scheduler'] = {
                    'warmup': (warm.state_dict() if warm else None),
                    'main': (main_sched.state_dict() if main_sched else None),
                }
                print(f"DEBUG: Initial scheduler state - warmup: {warm.last_epoch if warm else 'None'}/{warm.total_iters if warm else 'None'}")
            else:
                state['scheduler'] = scheduler.state_dict()
                print(f"DEBUG: Initial regular scheduler state")
        except Exception as e:
            print(f"WARNING: Failed to save initial scheduler state: {e}")

    for epoch in range(start_epoch + 1, args.total_epoch):
        start = time.time()
        
        # Print current LR at the beginning of each epoch
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\n=== Epoch {epoch} Started ===")
        print(f"Current Learning Rate: {current_lr}")
        
        # Manual learning rate decay for StepLR at the exact step_size epoch
        if args.scheduler == 'steplr' and epoch == args.steplr_stepsize:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= args.steplr_gamma
            print(f"DEBUG: Manual StepLR decay applied at epoch {epoch}. New LR: {optimizer.param_groups[0]['lr']}")

        # Warmup LR recorder for plotting once when warmup completes
        if epoch == (start_epoch + 1):
            warmup_recorder = {'lrs': [], 'done_saved': False}

        # Reset peak GPU memory statistics before training
        if device == 'cuda':
            torch.cuda.reset_peak_memory_stats()

        train_acc = train(
            net,
            epoch,
            device,
            train_loader,
            optimizer,
            criterion,
            args,
            writer,
            scheduler,
            warmup_recorder,
            ckpt_name
        )

        # Print optimizer-state and peak GPU memory after training
        if device == 'cuda':
            torch.cuda.synchronize()
            print_optimizer_memory(optimizer)
            print_gpu_memory()

        test_acc = test(
            net,
            device,
            test_loader,
            criterion,
            writer,
            epoch
        )
        
        
        if denom_csv is not None:
            denom_csv.log(optimizer, epoch)    
        end = time.time()
        print('Time: {}'.format(end - start))

        if scheduler:
            # Print current LR before step
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch}: LR before scheduler.step(): {current_lr}")
            
            # Log learning rate to TensorBoard after scheduler step
            writer.add_scalar('Learning_Rate_After_Scheduler', current_lr, epoch)
            
            if args.scheduler in ['warmupsteplr', 'warmupmultisteplr']:
                # After iterations, step the main StepLR per-epoch
                main_sched = scheduler.get('main')
                if main_sched:
                    main_sched.step()
                    new_lr = optimizer.param_groups[0]['lr']
                    print(f"Epoch {epoch}: LR after warmupsteplr main scheduler.step(): {new_lr}")
                    # Log learning rate after scheduler step
                    writer.add_scalar('Learning_Rate_After_Step', new_lr, epoch)
            elif args.scheduler not in ['cycliclr', 'linearlr']:
                scheduler.step()
                new_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch}: LR after scheduler.step(): {new_lr}")
                # Log learning rate after scheduler step
                writer.add_scalar('Learning_Rate_After_Step', new_lr, epoch)
            
            # Update scheduler state in state dictionary after step
            if isinstance(scheduler, dict):
                warm = scheduler.get('warmup')
                main_sched = scheduler.get('main')
                state['scheduler'] = {
                    'warmup': (warm.state_dict() if warm else None),
                    'main': (main_sched.state_dict() if main_sched else None),
                }
                # Debug: print scheduler state after update
                if warm:
                    print(f"Updated warmup state: last_epoch={warm.last_epoch}, total_iters={warm.total_iters}")
                if main_sched:
                    print(f"Updated main scheduler state: last_epoch={main_sched.last_epoch}")
            else:
                state['scheduler'] = scheduler.state_dict()

        # Save checkpoint based on save_interval
        should_save = False
        if args.save_interval > 0 and (epoch + 1) % args.save_interval == 0:
            should_save = True
            print('Saving checkpoint for epoch {}..'.format(epoch))
        elif args.save_interval == -1 and epoch == args.total_epoch - 1:
            should_save = True
            print('Saving checkpoint at final epoch {}..'.format(epoch))
        
        if should_save:
            state.update({
                'net': net.state_dict(),
                'acc': test_acc,
                'epoch': epoch,
                'optimizer': optimizer.state_dict(),
            })
            # Update scheduler state in checkpoint
            if scheduler:
                try:
                    if isinstance(scheduler, dict):
                        warm = scheduler.get('warmup')
                        main_sched = scheduler.get('main')
                        state['scheduler'] = {
                            'warmup': (warm.state_dict() if warm else None),
                            'main': (main_sched.state_dict() if main_sched else None),
                        }
                        print(f"DEBUG: Saved warmup scheduler state - warmup: {warm.last_epoch if warm else 'None'}/{warm.total_iters if warm else 'None'}")
                    else:
                        state['scheduler'] = scheduler.state_dict()
                        print(f"DEBUG: Saved regular scheduler state")
                except Exception as e:
                    print(f"WARNING: Failed to save scheduler state: {e}")
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, os.path.join('checkpoint', ckpt_name))

        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        if not os.path.isdir('curve'):
            os.mkdir('curve')
        torch.save({'train_acc': train_accuracies, 'test_acc': test_accuracies},
                   os.path.join('curve', ckpt_name))

    writer.close()


if __name__ == '__main__':
    main()
   