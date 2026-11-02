import math
import torch
from torch.optim.optimizer import Optimizer

class EmojiAbl(torch.optim.Optimizer):
    """Emoji optimizer with an adjustable residual-centering coefficient."""

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        weight_decouple=True,
        amsgrad=False,
        alphaa=0,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if len(betas) != 2:
            raise ValueError(f"Expected two beta values, got {betas}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= alphaa <= 1.0:
            raise ValueError(f"Invalid alphaa value: {alphaa}; expected a value in [0, 1]")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            weight_decouple=weight_decouple,
            amsgrad=amsgrad,
            alphaa=alphaa,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform one EmojiAbl optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            alpha = group["alphaa"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.detach()
                if grad.is_sparse:
                    raise RuntimeError(
                        "EmojiAbl does not support sparse gradients; "
                        "please consider SparseAdam instead."
                    )

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    state["exp_var"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    state["jerk"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    if group["amsgrad"]:
                        state["max_exp_avg_var"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )

                exp_avg = state["exp_avg"]
                exp_var = state["exp_var"]
                jerk = state["jerk"]
                if group["amsgrad"]:
                    max_exp_avg_var = state["max_exp_avg_var"]

                state["step"] += 1
                current_step = state["step"]

                # Apply weight decay (same ordering as the original Emoji code).
                if group["weight_decay"] != 0.0 and not group["weight_decouple"]:
                    grad = grad.add(p, alpha=group["weight_decay"])
                if group["weight_decay"] != 0.0 and group["weight_decouple"]:
                    p.mul_(1.0 - group["lr"] * group["weight_decay"])

                # Update first moment (m).
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)

                # Compute residual and its moving average (preserved from Emoji).
                d = grad - exp_avg
                exp_var.mul_(beta1).add_(d, alpha=1.0 - beta1)

                # Preserve the exact original Emoji expression when alpha == 1.
                # Skip centering work entirely when alpha == 0.
                if alpha == 0.0:
                    jerk.mul_(beta2).addcmul_(
                        d,
                        d,
                        value=1.0 - beta2,
                    )
                elif alpha == 1.0:
                    jerk.mul_(beta2).addcmul_(
                        d - exp_var,
                        d - exp_var,
                        value=1.0 - beta2,
                    )
                else:
                    centered_residual = d - alpha * exp_var
                    jerk.mul_(beta2).addcmul_(
                        centered_residual,
                        centered_residual,
                        value=1.0 - beta2,
                    )

                # Bias correction
                bias_correction1 = 1.0 - beta1**current_step
                bias_correction2 = 1.0 - beta2**current_step
                m_hat = exp_avg / bias_correction1
                j_hat = jerk / bias_correction2

                # Keep the denominator behavior of the original Emoji implementation.
                if group["amsgrad"]:
                    torch.max(max_exp_avg_var, jerk, out=max_exp_avg_var)
                    denom = max_exp_avg_var.sqrt().add_(group["eps"])
                else:
                    denom = j_hat.sqrt().add_(group["eps"])

                p.addcdiv_(m_hat, denom, value=-group["lr"])

        return loss
    
