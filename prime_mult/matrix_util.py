import torch
import torch.nn as nn


def create_linear_layer(lin_fn, num_inputs):
    """
    Turn a linear function into a linear layer.

    :param lin_fn: a linear function that takes (inputs, bias) and returns an
                   iterator of scalar torch Tensors.
    :return: an nn.Linear layer.
    """
    inputs = torch.randn(num_inputs, dtype=torch.float64).requires_grad_(True)
    bias = torch.ones((), dtype=torch.float64).requires_grad_(True)
    out = lin_fn(inputs, bias)
    layer = nn.Linear(num_inputs, len(out))
    for i, x in enumerate(out):
        inputs.grad = None
        bias.grad = None
        x.backward()
        if inputs.grad is not None:
            layer.weight.detach()[i] = inputs.grad.float()
        else:
            layer.weight.detach()[i].zero_()
        if bias.grad is not None:
            layer.bias.detach()[i] = bias.grad.float()
        else:
            layer.bias.detach()[i] = 0
    return layer
