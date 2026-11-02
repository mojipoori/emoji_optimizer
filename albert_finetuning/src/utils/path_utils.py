import datetime
import math

def generate_run_name(config):
    """
    Generates a descriptive run name based on hyperparameters with a timestamp for uniqueness.
    """
   
    task = config['task']
    
    lr = config['learning_rate']
    bsz = config['batch_size']
    albert_dr = config['albert_dropout']
    classifier_dr = config['classifier_dropout']
    ts = config['training_steps']
    ws = config['warmup_steps']
    msl = config['max_seq_length']
    weight_decay = config.get('weight_decay', 0.01)
    optimizer = config.get('optimizer', 'emoji')
                

    # Optimizer-specific hyperparameters
    opt_params = f"OPT{optimizer}"
    if optimizer == 'adam':
        opt_params += f"_B{config.get('adam_betas', [0.9, 0.999])[0]}-{config.get('adam_betas', [0.9, 0.999])[1]}"
        opt_params += f"_EPS{config.get('adam_eps', 1e-8)}"
        opt_params += f"_AMSG{config.get('adam_amsgrad', False)}"
    elif optimizer == 'adamw':
        opt_params += f"_B{config.get('adamw_betas', [0.9, 0.999])[0]}-{config.get('adamw_betas', [0.9, 0.999])[1]}"
        opt_params += f"_EPS{config.get('adamw_eps', 1e-8)}"
        opt_params += f"_AMSG{config.get('adamw_amsgrad', False)}"
    elif optimizer == 'sgd':
        opt_params += f"_MOM{config.get('sgd_momentum', 0.9)}"
        opt_params += f"_DAMP{config.get('sgd_dampening', 0.0)}"
        opt_params += f"_NEST{config.get('sgd_nesterov', False)}"
    elif optimizer == 'adabelief':
        opt_params += f"_B{config.get('adabelief_betas', [0.9, 0.999])[0]}-{config.get('adabelief_betas', [0.9, 0.999])[1]}"
        opt_params += f"_EPS{config.get('adabelief_eps', 1e-16)}"
        opt_params += f"_AMSG{config.get('adabelief_amsgrad', False)}"
        opt_params += f"_WD{config.get('adabelief_weight_decouple', True)}"
        opt_params += f"_RECT{config.get('adabelief_rectify', True)}"

    elif optimizer == 'emoji':
        opt_params += f"_B{config.get('emoji_betas', [0.9, 0.999])[0]}-{config.get('emoji_betas', [0.9, 0.999])[1]}"
        opt_params += f"_EPS{config.get('emoji_eps', 1e-8)}"
        opt_params += f"_AMSG{config.get('emoji_amsgrad', False)}"
        opt_params += f"_WD{config.get('emoji_weight_decouple', True)}"

    elif optimizer == 'yogi':
        opt_params += f"_B{config.get('yogi_betas', [0.9, 0.999])[0]}-{config.get('yogi_betas', [0.9, 0.999])[1]}"
        opt_params += f"_EPS{config.get('yogi_eps', 1e-3)}"
        opt_params += f"_IA{config.get('yogi_initial_accumulator', 1e-6)}"
        opt_params += f"_WD{config.get('yogi_weight_decay', 0)}"
    elif optimizer == 'radam':
        opt_params += f"_B{config.get('radam_betas', [0.9, 0.999])[0]}-{config.get('radam_betas', [0.9, 0.999])[1]}"
        opt_params += f"_EPS{config.get('radam_eps', 1e-8)}"
        opt_params += f"_WD{config.get('radam_weight_decay', 0)}"
        opt_params += f"_DTSGD{config.get('radam_degenerated_to_sgd', True)}"
    elif optimizer == 'msvag':
        opt_params += f"_B{config.get('msvag_betas', [0.9, 0.999])[0]}-{config.get('msvag_betas', [0.9, 0.999])[1]}"
        opt_params += f"_EPS{config.get('msvag_eps', 1e-8)}"
        opt_params += f"_WD{config.get('msvag_weight_decay', 0)}"
        opt_params += f"_AMSG{config.get('msvag_amsgrad', False)}"
    elif optimizer == 'fromage':
        opt_params += f"_LR{config.get('fromage_lr', 1e-3)}"
    elif optimizer == 'adabound':
        opt_params += f"_B{config.get('adabound_betas', [0.9, 0.999])[0]}-{config.get('adabound_betas', [0.9, 0.999])[1]}"
        opt_params += f"_EPS{config.get('adabound_eps', 1e-8)}"
        opt_params += f"_FLR{config.get('adabound_final_lr', 0.1)}"
        opt_params += f"_GAMMA{config.get('adabound_gamma', 1e-3)}"
        opt_params += f"_WD{config.get('adabound_weight_decay', 0)}"
        opt_params += f"_AMSBD{config.get('adabound_amsbound', False)}"
    elif optimizer == 'adaboundw':
        opt_params += f"_B{config.get('adaboundw_betas', [0.9, 0.999])[0]}-{config.get('adaboundw_betas', [0.9, 0.999])[1]}"
        opt_params += f"_EPS{config.get('adaboundw_eps', 1e-8)}"
        opt_params += f"_FLR{config.get('adaboundw_final_lr', 0.1)}"
        opt_params += f"_GAMMA{config.get('adaboundw_gamma', 1e-3)}"
        opt_params += f"_WD{config.get('adaboundw_weight_decay', 0)}"
        opt_params += f"_AMSBD{config.get('adaboundw_amsbound', False)}"
    elif optimizer == 'lion':
        opt_params += f"_LR{config.get('lion_lr', 1e-4)}"
        opt_params += f"_B{config.get('lion_betas', [0.9, 0.99])[0]}-{config.get('lion_betas', [0.9, 0.99])[1]}"
        opt_params += f"_WD{config.get('lion_weight_decay', 0.0)}"

    # Generate timestamp in format YYYYMMDD_HHMMSS
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    reproducible_tag_str = f"_SEED{config.get('seed', 'None')}" if config.get('reproducible_tag', False) else ""

    # Format the run name to include key hyperparameters, optimizer params, and timestamp
    
    # For Lion we omit the general LR and WD because Lion uses its own lion_lr / lion_weight_decay

    if optimizer != 'lion':
   
        run_name = (
            f"{task}_LR{lr}_BSZ{bsz}_GAS{config.get('gradient_accumulation_steps', 1)}_ADR{albert_dr}_CDR{classifier_dr}_TS{ts}_WS{ws}_MSL{msl}_WD{weight_decay}_{opt_params}{reproducible_tag_str}_{timestamp}"
        )
    else:
        run_name = (
            f"{task}_BSZ{bsz}_GAS{config.get('gradient_accumulation_steps', 1)}_ADR{albert_dr}_CDR{classifier_dr}_TS{ts}_WS{ws}_MSL{msl}_{opt_params}{reproducible_tag_str}_{timestamp}"
        )

           
        
    return run_name