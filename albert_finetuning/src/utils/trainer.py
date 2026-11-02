from transformers import Trainer, TrainingArguments
import torch.optim as optim
from src.optimizers import AdamW, AdaBelief, Emoji, Yogi, RAdam, Fromage, AdaBound, AdaBoundW, Lion
import os
import math

class CustomTrainer(Trainer):
    def __init__(self, *args, config=None, log_dir=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.log_dir = log_dir
    def create_optimizer(self):
        """
        Override the default optimizer creation to support custom optimizers.
        """
        optimizer_name = self.config.get('optimizer', 'emoji').lower()
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.get('lion_weight_decay', 0.0) if optimizer_name == 'lion' else self.config.get('weight_decay', 0.0),
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        if optimizer_name == 'adam':
            optimizer = optim.Adam(
                optimizer_grouped_parameters,
                lr=self.args.learning_rate,
                betas=self.config.get('adam_betas', [0.9, 0.999]),
                eps=self.config.get('adam_eps', 1e-8),
                weight_decay=self.config.get('weight_decay', 0.01),
                amsgrad=self.config.get('adam_amsgrad', False),
                foreach=self.config.get('adam_foreach', None),
                maximize=self.config.get('adam_maximize', False),
                capturable=self.config.get('adam_capturable', False),
                differentiable=self.config.get('adam_differentiable', False),
                fused=self.config.get('adam_fused', None)
            )
        elif optimizer_name == 'adamw':
            optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=self.args.learning_rate,
                betas=self.config.get('adamw_betas', [0.9, 0.999]),
                eps=self.config.get('adamw_eps', 1e-8),
                weight_decay=self.config.get('weight_decay', 0.01),
                amsgrad=self.config.get('adamw_amsgrad', False)
            )
        elif optimizer_name == 'sgd':
            optimizer = optim.SGD(
                optimizer_grouped_parameters,
                lr=self.args.learning_rate,
                momentum=self.config.get('sgd_momentum', 0.9),
                weight_decay=self.config.get('weight_decay', 0.01),
                dampening=self.config.get('sgd_dampening', 0.0),
                nesterov=self.config.get('sgd_nesterov', False)
            )
        elif optimizer_name == 'adabelief':
            optimizer = AdaBelief(
                optimizer_grouped_parameters,
                lr=self.args.learning_rate,
                betas=self.config.get('adabelief_betas', [0.9, 0.999]),
                eps=self.config.get('adabelief_eps', 1e-16),
                weight_decay=self.config.get('weight_decay', 0.01),
                amsgrad=self.config.get('adabelief_amsgrad', False),
                weight_decouple=self.config.get('adabelief_weight_decouple', True),
                fixed_decay=self.config.get('adabelief_fixed_decay', False),
                rectify=self.config.get('adabelief_rectify', True),
                degenerated_to_sgd=self.config.get('adabelief_degenerated_to_sgd', True),
                print_change_log=True,
            )
        elif optimizer_name == 'emoji':
            optimizer = Emoji(
                optimizer_grouped_parameters,
                lr=self.args.learning_rate,
                betas=self.config.get('emoji_betas', (0.9, 0.999)),
                eps=self.config.get('emoji_eps', 1e-8),
                weight_decay=self.config.get('weight_decay', 0.01),
                weight_decouple=self.config.get('emoji_weight_decouple', True),
                amsgrad=self.config.get('emoji_amsgrad', False),
           )
        elif optimizer_name == 'yogi':
            optimizer = Yogi(
                optimizer_grouped_parameters,
                lr=self.args.learning_rate,
                betas=self.config.get('yogi_betas', (0.9, 0.999)),
                eps=self.config.get('yogi_eps', 1e-3),
                initial_accumulator=self.config.get('yogi_initial_accumulator', 1e-6),
                weight_decay=self.config.get('weight_decay', 0)
            )
        elif optimizer_name == 'radam':
            optimizer = RAdam(
                optimizer_grouped_parameters,
                lr=self.args.learning_rate,
                betas=self.config.get('radam_betas', (0.9, 0.999)),
                eps=self.config.get('radam_eps', 1e-8),
                weight_decay=self.config.get('weight_decay', 0),
                degenerated_to_sgd=self.config.get('radam_degenerated_to_sgd', True)
            )

        elif optimizer_name == 'fromage':
            optimizer = Fromage(
                optimizer_grouped_parameters,
                lr=self.config.get('fromage_lr', 1e-3)
            )
        elif optimizer_name == 'adabound':
            optimizer = AdaBound(
                optimizer_grouped_parameters,
                lr=self.args.learning_rate,
                betas=self.config.get('adabound_betas', (0.9, 0.999)),
                final_lr=self.config.get('adabound_final_lr', 0.1),
                gamma=self.config.get('adabound_gamma', 1e-3),
                eps=self.config.get('adabound_eps', 1e-8),
                weight_decay=self.config.get('weight_decay', 0),
                amsbound=self.config.get('adabound_amsbound', False)
            )
        elif optimizer_name == 'adaboundw':
            optimizer = AdaBoundW(
                optimizer_grouped_parameters,
                lr=self.args.learning_rate,
                betas=self.config.get('adaboundw_betas', (0.9, 0.999)),
                final_lr=self.config.get('adaboundw_final_lr', 0.1),
                gamma=self.config.get('adaboundw_gamma', 1e-3),
                eps=self.config.get('adaboundw_eps', 1e-8),
                weight_decay=self.config.get('weight_decay', 0),
                amsbound=self.config.get('adaboundw_amsbound', False)
            )
        elif optimizer_name == 'lion':
            optimizer = Lion(
                optimizer_grouped_parameters,
                lr=self.config.get('lion_lr', 1e-4),
                betas=self.config.get('lion_betas', (0.9, 0.99)),
                weight_decay=self.config.get('lion_weight_decay', 0.0)
            )
        else:
            raise ValueError(f"Optimizer {optimizer_name} not supported. Choose from 'adam', 'adamw', 'sgd', 'adabelief', 'emoji', 'yogi', 'radam', 'msvag', 'fromage', 'adabound', 'adaboundw', 'lion'.")

        self.optimizer = optimizer
        return self.optimizer
