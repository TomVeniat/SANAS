import logging

import torch
from torch.optim.lr_scheduler import StepLR

logger = logging.getLogger(__name__)


def optim_config(ex):
    @ex.config
    def basic_config():
        lr_pred = 1e-3
        lr_arch = 1e-3
        wd_pred = 1e-5
        wd_arch = 0

    @ex.named_config
    def adam():
        optim_name = 'adam'
        betas = (0.9, 0.999)

    @ex.named_config
    def sgd():
        optim_name = 'sgd'
        momentum_pred = 0.9
        momentum_arch = 0
        step_size = 500
        step_fact = .1

    @ex.capture
    def create_sgd_optim(params, lr_arch, lr_pred, momentum_arch, momentum_pred, wd_arch, wd_pred, step_size, step_fact):
        params['arch']['lr'] = lr_arch
        params['arch']['momentum'] = momentum_arch
        params['arch']['weight_decay'] = wd_arch

        params['pred']['lr'] = lr_pred
        params['pred']['momentum'] = momentum_pred
        params['pred']['weight_decay'] = wd_pred

        optimizer = torch.optim.SGD(params.values())
        schedulder = StepLR(optimizer, step_size=step_size, gamma=step_fact)
        return optimizer, schedulder

    @ex.capture
    def create_adam_optim(params, lr_arch, lr_pred, betas,  wd_arch, wd_pred):
        params['arch']['lr'] = lr_arch
        params['arch']['weight_decay'] = wd_arch

        params['pred']['lr'] = lr_pred
        params['pred']['weight_decay'] = wd_pred

        optimizer = torch.optim.Adam(params=params.values(), betas=betas)
        schedulder = None
        return optimizer, schedulder

    optim_funcs = {
        'sgd': create_sgd_optim,
        'adam': create_adam_optim,
    }

    @ex.capture
    def select_optim(optim_name, **kwargs):
        return optim_funcs[optim_name](**kwargs)

    return select_optim
