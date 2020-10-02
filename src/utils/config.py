import torch
import inspect

def get_optim(parameters, cfg):
    try:
        import apex
    except ImportError:
        print('Please install apex using...')
    if hasattr(torch.optim, cfg.name):
        cls = getattr(torch.optim, cfg.name)
    elif hasattr(apex.optimizers, cfg.name):
        cls = getattr(apex.optimizers, cfg.name)
    else:
        raise ValueError(f'optimizer {optimizer} is invalid.')
    kwargs = {}
    for k, v in cfg.items():
        if k in inspect.signature(cls.__init__).parameters:
            kwargs[k] = v
    return cls(parameters, **kwargs)
