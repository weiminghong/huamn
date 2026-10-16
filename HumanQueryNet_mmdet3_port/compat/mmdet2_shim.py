from functools import wraps

import torch


def force_fp32(apply_to=None, out_fp16=False):
    """Minimal mmcv.runner.force_fp32 replacement for MMEngine ports.

    The original decorator mainly casts selected fp16 tensors to fp32 under
    mixed precision. For this port smoke path we keep behavior conservative:
    selected Tensor arguments are cast to fp32 and outputs are left unchanged
    unless out_fp16=True.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if apply_to:
                # Methods use args[0] as self; map positional names from code is
                # expensive here, so only kwargs are cast in this compatibility shim.
                for name in apply_to:
                    value = kwargs.get(name)
                    if isinstance(value, torch.Tensor) and value.is_floating_point():
                        kwargs[name] = value.float()
            result = func(*args, **kwargs)
            if out_fp16 and isinstance(result, torch.Tensor) and result.is_floating_point():
                return result.half()
            return result
        return wrapper

    return decorator


def jit(*jit_args, **jit_kwargs):
    def decorator(func):
        return func
    if jit_args and callable(jit_args[0]) and len(jit_args) == 1 and not jit_kwargs:
        return jit_args[0]
    return decorator


def collate(batch, samples_per_gpu=1):
    from mmengine.dataset import pseudo_collate
    return pseudo_collate(batch)


class DataContainer:
    def __init__(self, data, stack=False, padding_value=0, cpu_only=False, pad_dims=2):
        self.data = data
        self.stack = stack
        self.padding_value = padding_value
        self.cpu_only = cpu_only
        self.pad_dims = pad_dims

    def __repr__(self):
        return f'DataContainer(data={type(self.data).__name__}, stack={self.stack}, cpu_only={self.cpu_only})'
