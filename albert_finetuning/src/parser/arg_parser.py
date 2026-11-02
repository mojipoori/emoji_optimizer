import argparse
import json
import os
import math

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune ALBERT base v2 for downstream tasks")
    parser.add_argument('--task', type=str, required=True, 
                        choices=['cola', 'sts', 'sst2', 'mnli', 'qnli', 'qqp', 'rte', 'mrpc', 'wnli', 
                                 'squad_v1', 'squad_v2', 'race'],
                        help='Task to fine-tune')
    parser.add_argument('--config', type=str, default=None, 
                        help='Path to config file (overrides default config)')
    parser.add_argument('--learning_rate', type=float, default=None, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--albert_dropout', type=float, default=None, help='ALBERT dropout rate')
    parser.add_argument('--classifier_dropout', type=float, default=None, help='Classifier dropout rate')
    parser.add_argument('--training_steps', type=int, default=None, help='Training steps')
    parser.add_argument('--warmup_steps', type=int, default=None, help='Warmup steps')
    parser.add_argument('--max_seq_length', type=int, default=None, help='Maximum sequence length')
    parser.add_argument('--weight_decay', type=float, default= 0.01, help='Weight decay (L2 penalty) for all optimizers')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Number of updates steps to accumulate before performing a backward/update pass.')
    parser.add_argument('--optimizer', type=str, default=None, 
                        choices=['adam', 'adamw', 'sgd', 'adabelief', 'emoji', 'yogi', 'radam', 'msvag', 'fromage', 'adabound', 'adaboundw', 'lion'], 
                        help='Optimizer to use (adam, adamw, sgd, adabelief, emoji, yogi, radam, msvag, fromage, adabound, adaboundw, lion)')
    
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
    parser.add_argument(
        '--adamw_amsgrad',
        action='store_true',
        default=False,
        help='Use AMSGrad variant',
    )
    # SGD-specific arguments
    parser.add_argument('--sgd_momentum', type=float, default=0.9, 
                        help='SGD momentum factor')
    parser.add_argument('--sgd_dampening', type=float, default=0.0, 
                        help='SGD dampening for momentum')
    parser.add_argument('--sgd_nesterov', type=bool, default=False, 
                        help='Use Nesterov momentum for SGD')
    
    # AdaBelief-specific arguments
    parser.add_argument('--adabelief_betas', type=float, nargs=2, default=[0.9, 0.999], 
                        help='AdaBelief betas (beta1, beta2)')
    parser.add_argument('--adabelief_eps', type=float, default=1e-8, 
                        help='AdaBelief epsilon for numerical stability')
    parser.add_argument(
        '--adabelief_amsgrad',
        action='store_true',
        default=False,
        help='Use AMSGrad variant',
    )
    parser.add_argument('--adabelief_weight_decouple', type=bool, default=True, 
                        help='Use decoupled weight decay for AdaBelief')
    parser.add_argument('--adabelief_fixed_decay', type=bool, default=False, 
                        help='Use fixed weight decay for AdaBelief')
    parser.add_argument('--adabelief_rectify', type=bool, default=False, 
                        help='Use rectified update for AdaBelief')
    parser.add_argument('--adabelief_degenerated_to_sgd', type=bool, default=False, 
                        help='Degenerate to SGD for high variance in AdaBelief')
    
    # emoji-specific arguments
    parser.add_argument('--emoji_betas', type=float, nargs=2, default=[0.9, 0.999],
                        help='emoji betas (beta1, beta2)')
    parser.add_argument('--emoji_eps', type=float, default=1e-8,
                        help='emoji epsilon for numerical stability')
    parser.add_argument('--emoji_weight_decouple', type=bool, default=True,
                        help='Use decoupled weight decay for emoji')
    parser.add_argument(
        '--emoji_amsgrad',
        action='store_true',
        default=False,
        help='Use AMSGrad variant for Emoji',
    )

    # Yogi-specific arguments
    parser.add_argument('--yogi_betas', type=float, nargs=2, default=[0.9, 0.999],
                        help='Yogi betas (beta1, beta2)')
    parser.add_argument('--yogi_eps', type=float, default=1e-3,
                        help='Yogi epsilon for numerical stability')
    parser.add_argument('--yogi_initial_accumulator', type=float, default=1e-6,
                        help='Yogi initial accumulator value')
    parser.add_argument('--yogi_weight_decay', type=float, default=0.01,
                        help='Yogi weight decay (L2 penalty)')
    
    # RAdam-specific arguments
    parser.add_argument('--radam_betas', type=float, nargs=2, default=[0.9, 0.999],
                        help='RAdam betas (beta1, beta2)')
    parser.add_argument('--radam_eps', type=float, default=1e-8,
                        help='RAdam epsilon for numerical stability')
    parser.add_argument('--radam_weight_decay', type=float, default=0.0,
                        help='RAdam weight decay (L2 penalty)')
    parser.add_argument('--radam_degenerated_to_sgd', type=bool, default=True,
                        help='RAdam degenerated to SGD for high variance')

    # MSVAG-specific arguments
    parser.add_argument('--msvag_betas', type=float, nargs=2, default=[0.9, 0.999],
                        help='MSVAG betas (beta1, beta2)')
    parser.add_argument('--msvag_eps', type=float, default=1e-8,
                        help='MSVAG epsilon for numerical stability')
    parser.add_argument('--msvag_weight_decay', type=float, default=0.0,
                        help='MSVAG weight decay (L2 penalty)')
    parser.add_argument('--msvag_amsgrad', type=bool, default=False,
                        help='Use AMSGrad variant for MSVAG')

    # Fromage-specific arguments
    parser.add_argument('--fromage_lr', type=float, default=1e-3,
                        help='Fromage learning rate')
    
    # AdaBound-specific arguments
    parser.add_argument('--adabound_betas', type=float, nargs=2, default=[0.9, 0.999],
                        help='AdaBound betas (beta1, beta2)')
    parser.add_argument('--adabound_eps', type=float, default=1e-8,
                        help='AdaBound epsilon for numerical stability')
    parser.add_argument('--adabound_final_lr', type=float, default=0.1,
                        help='AdaBound final learning rate')
    parser.add_argument('--adabound_gamma', type=float, default=1e-3,
                        help='AdaBound convergence speed of the bound functions')
    parser.add_argument('--adabound_weight_decay', type=float, default=0.01,
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
    parser.add_argument('--lion_weight_decay', type=float, default=0.1, help='Lion weight decay')

    parser.add_argument('--reproducible_tag', type=str, default=True,
                        help='A tag to identify reproducible runs. Default is None.')
    parser.add_argument('--seed', type=int, default=1993,
                        help='Random seed for reproducibility. Default is None (no fixed seed).')

    # Arguments for resuming training
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                        help='Path to a checkpoint directory to resume training from.')
    parser.add_argument('--new_max_steps', type=int, default=None,
                        help='New total number of training steps when resuming from a checkpoint.')

    args = parser.parse_args()

    # Load default config if not provided
    if args.config is None:
        config_path = os.path.join('configs', f'{args.task}_config.json')
    else:
        config_path = args.config

    config = load_config(config_path)

    # Override config with command-line arguments if provided
    if args.learning_rate is not None:
        config['learning_rate'] = args.learning_rate
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.albert_dropout is not None:
        config['albert_dropout'] = args.albert_dropout
    if args.classifier_dropout is not None:
        config['classifier_dropout'] = args.classifier_dropout
    if args.training_steps is not None:
        config['training_steps'] = args.training_steps
    if args.warmup_steps is not None:
        config['warmup_steps'] = args.warmup_steps
    if args.max_seq_length is not None:
        config['max_seq_length'] = args.max_seq_length
    if args.weight_decay is not None:
        config['weight_decay'] = args.weight_decay
    if args.gradient_accumulation_steps is not None:
        config['gradient_accumulation_steps'] = args.gradient_accumulation_steps
    if args.optimizer is not None:
        config['optimizer'] = args.optimizer

    # Apply 10x learning rate multiplier for Yogi if not specified by command line
    if config.get('optimizer') == 'yogi' and args.learning_rate is None:
        config['learning_rate'] *= 10.0

    # Override optimizer-specific arguments
    if args.adam_betas is not None:
        config['adam_betas'] = args.adam_betas
    if args.adam_eps is not None:
        config['adam_eps'] = args.adam_eps
    if args.adam_amsgrad is not None:
        config['adam_amsgrad'] = args.adam_amsgrad
    if args.adam_foreach is not None:
        config['adam_foreach'] = args.adam_foreach
    if args.adam_maximize is not None:
        config['adam_maximize'] = args.adam_maximize
    if args.adam_capturable is not None:
        config['adam_capturable'] = args.adam_capturable
    if args.adam_differentiable is not None:
        config['adam_differentiable'] = args.adam_differentiable
    if args.adam_fused is not None:
        config['adam_fused'] = args.adam_fused

    if args.adamw_betas is not None:
        config['adamw_betas'] = args.adamw_betas
    if args.adamw_eps is not None:
        config['adamw_eps'] = args.adamw_eps
    if args.adamw_amsgrad is not None:
        config['adamw_amsgrad'] = args.adamw_amsgrad

    if args.sgd_momentum is not None:
        config['sgd_momentum'] = args.sgd_momentum
    if args.sgd_dampening is not None:
        config['sgd_dampening'] = args.sgd_dampening
    if args.sgd_nesterov is not None:
        config['sgd_nesterov'] = args.sgd_nesterov

    if args.adabelief_betas is not None:
        config['adabelief_betas'] = args.adabelief_betas
    if args.adabelief_eps is not None:
        config['adabelief_eps'] = args.adabelief_eps
    if args.adabelief_amsgrad is not None:
        config['adabelief_amsgrad'] = args.adabelief_amsgrad
    if args.adabelief_weight_decouple is not None:
        config['adabelief_weight_decouple'] = args.adabelief_weight_decouple
    if args.adabelief_fixed_decay is not None:
        config['adabelief_fixed_decay'] = args.adabelief_fixed_decay
    if args.adabelief_rectify is not None:
        config['adabelief_rectify'] = args.adabelief_rectify
    if args.adabelief_degenerated_to_sgd is not None:
        config['adabelief_degenerated_to_sgd'] = args.adabelief_degenerated_to_sgd

    if args.emoji_betas is not None:
        config['emoji_betas'] = args.emoji_betas
    if args.emoji_eps is not None:
        config['emoji_eps'] = args.emoji_eps
    if args.emoji_weight_decouple is not None:
        config['emoji_weight_decouple'] = args.emoji_weight_decouple
    if args.emoji_amsgrad is not None:
        config['emoji_amsgrad'] = args.emoji_amsgrad
    

    if args.yogi_betas is not None:
        config['yogi_betas'] = args.yogi_betas
    if args.yogi_eps is not None:
        config['yogi_eps'] = args.yogi_eps
    if args.yogi_initial_accumulator is not None:
        config['yogi_initial_accumulator'] = args.yogi_initial_accumulator
    if args.yogi_weight_decay is not None:
        config['yogi_weight_decay'] = args.yogi_weight_decay

    if args.radam_betas is not None:
        config['radam_betas'] = args.radam_betas
    if args.radam_eps is not None:
        config['radam_eps'] = args.radam_eps
    if args.radam_weight_decay is not None:
        config['radam_weight_decay'] = args.radam_weight_decay
    if args.radam_degenerated_to_sgd is not None:
        config['radam_degenerated_to_sgd'] = args.radam_degenerated_to_sgd

    if args.msvag_betas is not None:
        config['msvag_betas'] = args.msvag_betas
    if args.msvag_eps is not None:
        config['msvag_eps'] = args.msvag_eps
    if args.msvag_weight_decay is not None:
        config['msvag_weight_decay'] = args.msvag_weight_decay
    if args.msvag_amsgrad is not None:
        config['msvag_amsgrad'] = args.msvag_amsgrad

    if args.fromage_lr is not None:
        config['fromage_lr'] = args.fromage_lr

    if args.adabound_betas is not None:
        config['adabound_betas'] = args.adabound_betas
    if args.adabound_eps is not None:
        config['adabound_eps'] = args.adabound_eps
    if args.adabound_final_lr is not None:
        config['adabound_final_lr'] = args.adabound_final_lr
    if args.adabound_gamma is not None:
        config['adabound_gamma'] = args.adabound_gamma
    if args.adabound_weight_decay is not None:
        config['adabound_weight_decay'] = args.adabound_weight_decay
    if args.adabound_amsbound is not None:
        config['adabound_amsbound'] = args.adabound_amsbound

    if args.adaboundw_betas is not None:
        config['adaboundw_betas'] = args.adaboundw_betas
    if args.adaboundw_eps is not None:
        config['adaboundw_eps'] = args.adaboundw_eps
    if args.adaboundw_final_lr is not None:
        config['adaboundw_final_lr'] = args.adaboundw_final_lr
    if args.adaboundw_gamma is not None:
        config['adaboundw_gamma'] = args.adaboundw_gamma
    if args.adaboundw_weight_decay is not None:
        config['adaboundw_weight_decay'] = args.adaboundw_weight_decay
    if args.adaboundw_amsbound is not None:
        config['adaboundw_amsbound'] = args.adaboundw_amsbound

    if args.lion_lr is not None:
        config['lion_lr'] = args.lion_lr
    if args.lion_betas is not None:
        config['lion_betas'] = args.lion_betas
    if args.lion_weight_decay is not None:
        config['lion_weight_decay'] = args.lion_weight_decay

    if args.reproducible_tag is not None:
        config['reproducible_tag'] = args.reproducible_tag
    if args.seed is not None:
        config['seed'] = args.seed


    return config, args

if __name__ == "__main__":
    config, task = parse_args()
    print(f"Loaded config for task {task}: {config}")