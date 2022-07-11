import math
import torch as th
import numpy as np
from torch.optim.optimizer import Optimizer, required

"""
Liyuan Liu , Haoming Jiang, Pengcheng He, Weizhu Chen, Xiaodong Liu, Jianfeng Gao, and Jiawei Han (2020). 
On the Variance of the Adaptive Learning Rate and Beyond. the Eighth International Conference on Learning 
Representations.
"""
class RAdam(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-16, weight_decay=0, degenerated_to_sgd=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        
        self.degenerated_to_sgd = degenerated_to_sgd
        if isinstance(params, (list, tuple)) and len(params) > 0 and isinstance(params[0], dict):
            for param in params:
                if 'betas' in param and (param['betas'][0] != betas[0] or param['betas'][1] != betas[1]):
                    param['buffer'] = [[None, None, None] for _ in range(10)]
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, buffer=[[None, None, None] for _ in range(10)])
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = th.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = th.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value = 1 - beta2)
                exp_avg.mul_(beta1).add_(grad, alpha = 1 - beta1)

                state['step'] += 1
                buffered = group['buffer'][int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    elif self.degenerated_to_sgd:
                        step_size = 1.0 / (1 - beta1 ** state['step'])
                    else:
                        step_size = -1
                    buffered[2] = step_size

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(exp_avg, denom, value = -step_size * group['lr'])
                    p.data.copy_(p_data_fp32)
                elif step_size > 0:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
                    p_data_fp32.add_(-step_size * group['lr'], exp_avg)
                    p.data.copy_(p_data_fp32)

        return loss


class SDRMSprop(Optimizer):

    def __init__(self, params, lr=1e-3, alpha=0.99, beta=0.9, eps=1e-8, weight_decay=0):

        defaults = dict(lr=lr, alpha=alpha, beta=beta, eps=eps, weight_decay=weight_decay)
        super(SDRMSprop, self).__init__(params, defaults)

    def reset_state(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['v'] = th.zeros_like(p, memory_format=th.preserve_format)
                state['s'] = th.zeros_like(p, memory_format=th.preserve_format)

    @th.no_grad()
    def step(self, debug=False):

        for group in self.param_groups:

            a = group['alpha']
            b = group['beta']
            lr  = group['lr']
            eps = group['eps']

            for p in group['params']:
                if p.grad is not None:

                    g = p.grad

                    state = self.state[p]

                    # Lazy state initialization
                    if len(state) == 0:
                        state['v'] = th.zeros_like(p, memory_format=th.preserve_format)
                        state['s'] = th.zeros_like(p, memory_format=th.preserve_format)

                    state['v'] = a * state['v'] + (1 - a) * g**2
                    state['s'] = b * state['s'] + (1 - b) * th.sign(g)

                    _v = state['v']
                    _s = state['s']


                    p.add_(-1 * lr * _s**2 * g / (th.sqrt(_v) + eps))



class SDAdam(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999, 0.9), eps=1e-8, weight_decay=0):

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(SDAdam, self).__init__(params, defaults)

    def reset_state(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step']  = 0
                state['m']     = th.zeros_like(p, memory_format=th.preserve_format)
                state['v']     = th.zeros_like(p, memory_format=th.preserve_format)
                state['s']     = th.zeros_like(p, memory_format=th.preserve_format)

    @th.no_grad()
    def step(self, debug=False):

        for group in self.param_groups:

            b1, b2, b3 = group['betas']
            lr  = group['lr']
            eps = group['eps']

            for p in group['params']:
                if p.grad is not None:

                    g = p.grad

                    state = self.state[p]

                    # Lazy state initialization
                    if len(state) == 0:
                        state['step']  = 0
                        state['m']     = th.zeros_like(p, memory_format=th.preserve_format)
                        state['v']     = th.zeros_like(p, memory_format=th.preserve_format)
                        state['s']     = th.zeros_like(p, memory_format=th.preserve_format)

                    state['step'] += 1

                    state['m'] = b1 * state['m'] + (1 - b1) * g
                    state['v'] = b2 * state['v'] + (1 - b2) * g**2
                    state['s'] = b3 * state['s'] + (1 - b3) * th.sign(g)

                    _m = state['m']     / (1 - b1**state['step'])
                    _v = state['v']     / (1 - b2**state['step'])
                    _s = state['s']     / (1 - b3**state['step'])


                    p.add_(-1 * lr * _s**2 * _m / (th.sqrt(_v) + eps))

class SDAMSGrad(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999, 0.9), eps=1e-8, weight_decay=0):

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(SDAMSGrad, self).__init__(params, defaults)

    def reset_state(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step']  = 0
                state['m']     = th.zeros_like(p, memory_format=th.preserve_format)
                state['v']     = th.zeros_like(p, memory_format=th.preserve_format)
                state['s']     = th.zeros_like(p, memory_format=th.preserve_format)
                state['v_max'] = th.zeros_like(p, memory_format=th.preserve_format)

    @th.no_grad()
    def step(self, debug=False):

        for group in self.param_groups:

            b1, b2, b3 = group['betas']
            lr  = group['lr']
            eps = group['eps']

            for p in group['params']:
                if p.grad is not None:

                    g = p.grad

                    state = self.state[p]

                    # Lazy state initialization
                    if len(state) == 0:
                        state['step']  = 0
                        state['m']     = th.zeros_like(p, memory_format=th.preserve_format)
                        state['v']     = th.zeros_like(p, memory_format=th.preserve_format)
                        state['s']     = th.zeros_like(p, memory_format=th.preserve_format)
                        state['v_max'] = th.zeros_like(p, memory_format=th.preserve_format)

                    state['step'] += 1

                    state['m'] = b1 * state['m'] + (1 - b1) * g
                    state['v'] = b2 * state['v'] + (1 - b2) * g**2
                    state['s'] = b3 * state['s'] + (1 - b3) * th.sign(g)
                    state['v_max'] = th.max(state['v_max'], state['v'])

                    _m = state['m']     / (1 - b1**state['step'])
                    _v = state['v_max'] / (1 - b2**state['step'])
                    _s = state['s']     / (1 - b3**state['step'])


                    p.add_(-1 * lr * _s**2 * _m / (th.sqrt(_v) + eps))


