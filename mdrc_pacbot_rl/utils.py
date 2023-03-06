import torch
from torch import nn

def copy_params(src: nn.Module, dest: nn.Module):
    """
    Copies params from one model to another.
    """
    with torch.no_grad():
        for dest_, src_ in zip(dest.parameters(), src.parameters()):
            dest_.data.copy_(src_.data)


def init_orthogonal(src: nn.Module):
    """
    Initializes model weights orthogonally. This has been shown to greatly
    improve training efficiency.
    """
    with torch.no_grad():
        for param in src.parameters():
            if len(param.size()) >= 2:
                param.copy_(torch.nn.init.orthogonal_(param.data))