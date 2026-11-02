import torch
from torch.optim.sgd import SGD as TorchSGD, sgd as _functional_sgd


class SGDW(TorchSGD):
    """SGD with decoupled weight decay (AdamW-style).

    Identical API/behavior to torch.optim.SGD except weight decay is applied in
    a decoupled manner: parameters are scaled by (1 - lr * weight_decay)
    prior to the gradient step, rather than adding weight decay to gradients.
    """

    # Inherit __init__ from torch.optim.SGD for arg parity

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        with torch.no_grad():
            for group in self.param_groups:
                params_with_grad = []
                d_p_list = []
                momentum_buffer_list = []

                has_sparse_grad = self._init_group(group, params_with_grad, d_p_list, momentum_buffer_list)

                # Decoupled weight decay: p <- (1 - lr * wd) * p
                if group['weight_decay'] != 0:
                    scale = 1 - group['lr'] * group['weight_decay']
                    for p in params_with_grad:
                        p.data.mul_(scale)

                # Call standard functional SGD with weight_decay disabled
                _functional_sgd(params_with_grad,
                                d_p_list,
                                momentum_buffer_list,
                                weight_decay=0.0,
                                momentum=group['momentum'],
                                lr=group['lr'],
                                dampening=group['dampening'],
                                nesterov=group['nesterov'],
                                maximize=group['maximize'],
                                has_sparse_grad=has_sparse_grad,
                                foreach=False)

                # Update momentum buffers same as upstream SGD
                for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                    state = self.state[p]
                    state['momentum_buffer'] = momentum_buffer

        return loss

 