import math
import torch
from torch.optim.optimizer import Optimizer

class MojiC(Optimizer):
    r"""Implements MojiC Optimizer Algorithm.

    This optimizer extends Adam by incorporating a jerk term, tracking the
    change in the difference between the gradient and its moving average, with
    variance and jerk updates as specified. It also supports the rectify mechanism
    from RAdam for adaptive learning rate adjustment based on gradient variance.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float, float], optional): coefficients for computing
            running averages of gradient, variance, and jerk (default: (0.9, 0.9, 0.999))
        eps (float, optional): term added to the denominator for numerical
            stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        weight_decouple (bool, optional): whether to use decoupled weight decay (default: False)
        amsgrad (bool, optional): whether to use AMSGrad variant (default: False)
        add_eps_to_final_momentum (str, optional): mode for adding epsilon to jerk
            ('off', 'fixed', 'linear', 'delayed_fixed', 'scaled_by_jerk') (default: 'off')
        eps_warmup_steps (int, optional): number of steps for linear epsilon warmup (default: 0)
        eps_start_step (int, optional): step to start applying fixed epsilon (default: 0)
        rectify (bool, optional): whether to use RAdam's rectify mechanism (default: True)
        gamma_sensitivity (float, optional): Sensitivity factor γ for ratio feedback (default: 1.0)
        sign_momentum (bool, optional): Whether to use sign(m_hat) instead of m_hat in the standard update (default: False)
        padam (bool, optional): Whether to use Padam's power term for variance (default: False)
        p (float, optional): Power term for Padam (between 0 and 0.5, default: 0.25)
        y_param (float, optional): Y parameter for cosine annealing (default is pi)
        rectify_threshold (int, optional): N_sma threshold for rectified update (default: 5)

    Example:
        >>> optimizer = MojiC(model.parameters(), lr=0.001, rectify=True)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.9, 0.999), eps=1e-8, weight_decay=0, 
                 weight_decouple=False, amsgrad=False, add_eps_to_final_momentum='off', 
                 eps_warmup_steps=0, eps_start_step=0, rectify=True, degenerated_to_sgd=True, 
                 a_param=0.1, training_steps=0, gamma_sensitivity=0.5, sign_momentum=False, padam=False, p=0.53, 
                 y_param=math.pi, rectify_threshold=5, a_factor=1.0):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= betas[2] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 2: {betas[2]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not add_eps_to_final_momentum in ['off', 'fixed', 'linear', 'delayed_fixed', 'scaled_by_jerk', 'exponential_ramp', 'hyperbolic_growth', 'cosine']:
            raise ValueError(f"Invalid add_eps_to_final_momentum mode. Choose from 'off', 'fixed', 'linear', 'delayed_fixed', 'scaled_by_jerk', 'exponential_ramp', 'hyperbolic_growth', or 'cosine'.")
        if not 0 <= eps_warmup_steps:
            raise ValueError(f"Invalid eps_warmup_steps value: {eps_warmup_steps}")
        if not 0 <= eps_start_step:
            raise ValueError(f"Invalid eps_start_step value: {eps_start_step}")
        if not 0.0 <= a_param:
            raise ValueError(f"Invalid a_param value: {a_param}")
        if not 0.0 <= gamma_sensitivity:
            raise ValueError(f"Invalid gamma_sensitivity value: {gamma_sensitivity}")
        if not 0.0 < p <= 0.5:
            raise ValueError(f"Invalid p value: {p}. Must be between 0 and 0.5 (exclusive of 0).")
        if not 0.0 <= y_param:
            raise ValueError(f"Invalid y_param value: {y_param}")
        if not 0 <= rectify_threshold:
            raise ValueError(f"Invalid rectify_threshold value: {rectify_threshold}")
        if not 0.0 <= a_factor <= 1.0:
            raise ValueError(f"Invalid a_factor value: {a_factor}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, 
                        weight_decouple=weight_decouple, amsgrad=amsgrad, 
                        add_eps_to_final_momentum=add_eps_to_final_momentum, 
                        eps_warmup_steps=eps_warmup_steps, eps_start_step=eps_start_step, 
                        rectify=rectify, degenerated_to_sgd=degenerated_to_sgd, 
                        a_param=a_param, training_steps=training_steps,
                        gamma_sensitivity=gamma_sensitivity,
                        sign_momentum=sign_momentum,
                        padam=padam,
                        p=p,
                        y_param=y_param,
                        rectify_threshold=rectify_threshold,
                        a_factor=a_factor,
                        buffer=[[None, None, None] for _ in range(10)])
        super(MojiC, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(MojiC, self).__setstate__(state)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        "MojiC does not support sparse gradients, "
                        "please consider SparseAdam instead"
                    )

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_var'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['jerk'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if group['amsgrad']:
                        state['max_exp_avg_var'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg = state['exp_avg']
                exp_var = state['exp_var']
                jerk = state['jerk']
                beta1, beta2, beta3 = group['betas']
                if group['amsgrad']:
                    max_exp_avg_var = state['max_exp_avg_var']

                state['step'] += 1
                current_step = state['step']

                # Apply weight decay (before rectify, as in MojiC original)
                if group['weight_decay'] != 0 and not group['weight_decouple']:
                    grad = grad.add(p.data, alpha=group['weight_decay'])
                if group['weight_decay'] != 0 and group['weight_decouple']:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])

                # Update first moment (m)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Compute variance (s)
                d = grad - exp_avg
                exp_var.mul_(beta2).add_(d, alpha=1 - beta2)

                # Compute jerk (j)
                jerk.mul_(beta3).addcmul_(d - exp_var, d - exp_var, value=1 - beta3)

                # Determine effective epsilon
                effective_eps = 0.0
                if group['add_eps_to_final_momentum'] == 'fixed':
                    effective_eps = group['eps']
                elif group['add_eps_to_final_momentum'] == 'linear':
                    if group['eps_warmup_steps'] > 0 and current_step <= group['eps_warmup_steps']:
                        effective_eps = (current_step / group['eps_warmup_steps']) * group['eps']
                    else:
                        effective_eps = group['eps']
                elif group['add_eps_to_final_momentum'] == 'delayed_fixed':
                    if current_step >= group['eps_start_step']:
                        effective_eps = group['eps']
                elif group['add_eps_to_final_momentum'] == 'scaled_by_jerk':
                    effective_eps = 3 * jerk * group['eps']
                elif group['add_eps_to_final_momentum'] == 'exponential_ramp':
                    # effective_eps = eps * (-exp(-a.x) + 1)
                    effective_eps = group['eps'] * (-math.exp(-group['a_param'] * current_step) + 1)
                elif group['add_eps_to_final_momentum'] == 'hyperbolic_growth':
                    # effective_eps = eps * ((-1/(a.x+1)) + 1)
                    effective_eps = group['eps'] * ((-1 / (group['a_param'] * current_step + 1)) + 1)
                elif group['add_eps_to_final_momentum'] == 'cosine':
                    # 1/2cos(y/training_steps*t + pi) + 1/2
                    effective_eps = group['eps'] * (0.5 * math.cos(group['y_param'] / group['training_steps'] * current_step + math.pi) + 0.5)

                # Apply effective epsilon to jerk
                if group['add_eps_to_final_momentum'] != 'off':
                    jerk.add_(effective_eps) # This is an in-place operation, modifying 'jerk' directly.

                # Bias correction
                bias_correction1 = 1 - beta1 ** current_step
                bias_correction3 = 1 - beta3 ** current_step
                m_hat = exp_avg / bias_correction1
                j_hat = jerk / bias_correction3

                if group['amsgrad']:
                    torch.max(max_exp_avg_var, jerk, out=max_exp_avg_var)
                    if group['padam']:
                        denom = max_exp_avg_var.pow(group['p']).add_(group['eps'])
                    else:
                        denom = max_exp_avg_var.sqrt().add_(group['eps'])
                else:
                    if group['padam']:
                        denom = j_hat.pow(group['p']).add_(group['eps'])
                    else:
                        denom = j_hat.sqrt().add_(group['eps'])

                # Apply rectify mechanism if enabled
                if group['rectify']:
                    buffered = group['buffer'][int(state['step'] % 10)]
                    if state['step'] == buffered[0]:
                        N_sma, step_size = buffered[1], buffered[2]
                    else:
                        buffered[0] = state['step']
                        beta2_t = beta2 ** state['step']
                        N_sma_max = 2 / (1 - group['a_factor'] * beta2) - 1
                        N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                        buffered[1] = N_sma

                        # Compute step size for rectified update
                        if N_sma >= group['rectify_threshold']:
                            step_size = math.sqrt(
                                (1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * 
                                (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)
                            ) / (1 - beta1 ** state['step'])
                        elif group['degenerated_to_sgd']:
                            step_size = 1.0 / (1 - beta1 ** state['step'])  # SGD-style update
                        else:
                            step_size = -1
                        buffered[2] = step_size

                    if N_sma >= group['rectify_threshold']:
                        if group['amsgrad']:
                            torch.max(max_exp_avg_var, jerk, out=max_exp_avg_var)
                            if group['padam']:
                                denom = max_exp_avg_var.pow(group['p']).add_(group['eps'])
                            else:
                                denom = max_exp_avg_var.sqrt().add_(group['eps'])
                        else:
                            if group['padam']:
                                denom = j_hat.pow(group['p']).add_(group['eps'])
                            else:
                                denom = j_hat.sqrt().add_(group['eps'])
                        p.data.addcdiv_(m_hat, denom, value=-step_size * group['lr'])
                    elif step_size > 0:
                        p.data.add_(m_hat, alpha=-step_size * group['lr'])  # SGD-style update
                else:
                    # Standard MojiC update
                    p.data.addcdiv_(m_hat, denom, value=-group['lr'])

        return loss