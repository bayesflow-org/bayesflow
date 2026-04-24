import torch


def jit(fn):
    return torch.compile(fn)
