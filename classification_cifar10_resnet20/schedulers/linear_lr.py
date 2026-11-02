import torch.optim.lr_scheduler as lr_scheduler
import argparse


def add_linear_lr_args(parser: argparse.ArgumentParser):
    parser.add_argument('--lin_start_factor', type=float, default=0.0, help='Starting LR factor for LinearLR (0.0 for warmup, 1.0 for regular)')
    parser.add_argument('--lin_end_factor', type=float, default=1.0, help='Ending LR factor for LinearLR (ignored if lin_target_lr is set)')
    parser.add_argument('--lin_total_iters', type=int, default=200, help='Total iterations (batches) over which to linearly ramp')
    parser.add_argument('--lin_last_epoch', type=int, default=-1, help='Index of last iteration for LinearLR (batch-based)')
    parser.add_argument('--lin_target_lr', type=float, default=None, help='Absolute target LR at the end of ramp (overrides end_factor if set)')


def get_optimizer_lr_for_scheduler(args):
    """
    Get the appropriate learning rate parameter for the specified optimizer.
    This is the same logic as in main.py but for scheduler use.
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
        'adabound': args.lr,
        'adaboundw': args.lr,
        'yogi': args.yogi_lr,
        'lion': args.lion_lr,
    }
    
    if args.optim in lr_mapping:
        lr_value = lr_mapping[args.optim]
        print(f"DEBUG: LinearLR - Using LR parameter for {args.optim}: {lr_value}")
        return lr_value
    else:
        print(f"WARNING: LinearLR - Unknown optimizer {args.optim}, using default --lr: {args.lr}")
        return args.lr


def create_linear_lr_scheduler(optimizer, args):
    """
    Create a LinearLR scheduler for warmup or regular linear learning rate scheduling.
    
    This function handles two scenarios:
    1. Warmup: LR starts from a small value and increases to base_lr
    2. Regular: LR changes linearly between start_factor and end_factor
    
    Args:
        optimizer: PyTorch optimizer
        args: Command line arguments containing scheduler parameters
    
    Returns:
        LinearLR scheduler object
    """
    
    # Get the TARGET learning rate that warmup should reach
    # This should be the LR parameter that the user specified for their optimizer
    target_lr = get_optimizer_lr_for_scheduler(args)
    
    print(f"DEBUG: LinearLR scheduler - target_lr: {target_lr}")
    print(f"DEBUG: This target_lr is what warmup will reach at the end")
    
    # ALWAYS: The optimizer's LR parameter should be the end target LR
    # This is the fundamental rule: whatever LR user specifies should be reached at the end
    
    # Handle user-specified target LR (overrides end_factor)
    if args.lin_target_lr is not None:
        # User explicitly specified a different target LR
        # Calculate end_factor to reach that target
        end_factors = []
        for group in optimizer.param_groups:
            group_base_lr = group.get('initial_lr', group['lr'])
            # end_factor = target_lr / base_lr
            end_factors.append(args.lin_target_lr / max(group_base_lr, 1e-12))
        # Use first group's ratio for simplicity (PyTorch LinearLR uses scalar factors)
        end_factor = end_factors[0]
        print(f"DEBUG: User specified target LR: {args.lin_target_lr}, calculated end_factor: {end_factor}")
    else:
        # DEFAULT: Always reach the target_lr (which is the optimizer's LR parameter)
        # This is the most common case: warmup should reach the specified optimizer LR
        end_factor = 1.0  # This ensures we reach target_lr * 1.0 = target_lr
        print(f"DEBUG: DEFAULT: Will reach target_lr ({target_lr}) using end_factor = 1.0")

    # WARMUP SCENARIO: When start_factor < 1.0, we're doing warmup
    if args.lin_start_factor < 1.0:
        print(f"DEBUG: WARMUP MODE detected (start_factor = {args.lin_start_factor} < 1.0)")
        
        # FUNDAMENTAL RULE: The optimizer's LR parameter is ALWAYS the target end LR
        # In warmup mode, we start from a small value and reach the target_lr value
        
        # The end_factor is already calculated above to reach the target LR
        # No need to recalculate it here - just confirm the configuration
        
        print(f"DEBUG: WARMUP RULE: target_lr ({target_lr}) is the target end LR")
        print(f"DEBUG: Warmup will go from {target_lr * args.lin_start_factor:.6f} to {target_lr * end_factor:.6f}")
        
        # Print warmup configuration
        print(f"DEBUG: Warmup configuration:")
        print(f"  - start_factor: {args.lin_start_factor}")
        print(f"  - end_factor: {end_factor}")
        print(f"  - target_lr (optimizer LR): {target_lr}")
        print(f"  - total_iters: {args.lin_total_iters}")
        print(f"  - Start LR: {target_lr * args.lin_start_factor:.6f}")
        print(f"  - End LR: {target_lr * end_factor:.6f}")
        print(f"  - RULE: End LR = target_lr * end_factor = {target_lr} * {end_factor} = {target_lr * end_factor:.6f}")
        
        # CRITICAL: Set the optimizer's initial LR to the starting warmup value
        # This ensures the warmup starts from the correct value
        start_lr = target_lr * args.lin_start_factor
        for param_group in optimizer.param_groups:
            param_group['lr'] = start_lr  # Set to warmup start value
            param_group['initial_lr'] = target_lr  # Keep original target_lr for scheduler
        print(f"DEBUG: Set optimizer LR to warmup start value: {start_lr:.6f}")
        print(f"DEBUG: Target LR for warmup: {target_lr:.6f}")
        print(f"DEBUG: Start factor: {args.lin_start_factor}")
        print(f"DEBUG: Expected start LR: {target_lr * args.lin_start_factor:.6f}")

    # Create and return the LinearLR scheduler
    # This scheduler will interpolate LR between start_factor and end_factor over total_iters
    scheduler = lr_scheduler.LinearLR(
        optimizer,
        start_factor=args.lin_start_factor,  # Starting factor (e.g., 0.001 for warmup)
        end_factor=end_factor,               # Ending factor (e.g., 1.0 to reach target_lr)
        total_iters=args.lin_total_iters,    # Number of iterations for the ramp
        last_epoch=args.lin_last_epoch       # Starting epoch (-1 for beginning)
    )
    
    # CRITICAL: After creating the scheduler, ensure the optimizer LR is set correctly
    # This is because some optimizers (like AdaBelief) might change the LR during creation
    if args.lin_start_factor < 1.0:  # Warmup mode
        start_lr = target_lr * args.lin_start_factor
        for param_group in optimizer.param_groups:
            param_group['lr'] = start_lr  # Force set the correct start LR
        print(f"DEBUG: FORCED optimizer LR to warmup start value: {start_lr:.6f}")
    
    # Print final scheduler configuration
    print(f"DEBUG: LinearLR scheduler created:")
    print(f"  - start_factor: {scheduler.start_factor}")
    print(f"  - end_factor: {scheduler.end_factor}")
    print(f"  - total_iters: {scheduler.total_iters}")
    print(f"  - Formula: lr(step) = target_lr × [start_factor + (end_factor - start_factor) × (step / total_iters)]")
    print(f"  - Final LR will be: {target_lr} × {scheduler.end_factor} = {target_lr * scheduler.end_factor:.6f}")
    
    return scheduler


def append_linear_lr_params(args, name: str):
    name += f'-linT{args.lin_total_iters}'
    if args.lin_start_factor != 1.0:
        name += f'-linS{args.lin_start_factor}'
    # Prefer target lr tag if set
    if args.lin_target_lr is not None:
        name += f'-linTLR{args.lin_target_lr}'
    elif args.lin_end_factor != 1.0:
        name += f'-linE{args.lin_end_factor}'
    if args.lin_last_epoch != -1:
        name += f'-linLEpoch{args.lin_last_epoch}'
    return name