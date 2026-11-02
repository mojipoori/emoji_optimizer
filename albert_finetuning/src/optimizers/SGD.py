import torch
from torch.optim.sgd import SGD as TorchSGD, sgd as _functional_sgd


class SGD(TorchSGD):
    """SGD with standard L2 regularization (coupled weight decay).

    Identical API/behavior to torch.optim.SGD where weight decay is applied
    as a penalty added directly to the gradients prior to the momentum step,
    rather than scaling parameters directly.
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

                # Call standard functional SGD with standard L2 weight decay enabled
                _functional_sgd(params_with_grad,
                                d_p_list,
                                momentum_buffer_list,
                                weight_decay=group['weight_decay'],
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