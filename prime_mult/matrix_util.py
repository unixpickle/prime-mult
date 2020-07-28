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


def repeat_block_diagonal(linear_layer, repeats):
    num_outputs, num_inputs = linear_layer.weight.shape
    new_layer = nn.Linear(num_inputs * repeats, num_outputs * repeats)
    new_layer.weight.detach().zero_()
    new_layer.bias.detach().zero_()
    for i in range(repeats):
        new_layer.bias.detach()[i * num_outputs : (i + 1) * num_outputs].copy_(
            linear_layer.bias
        )
        new_layer.weight.detach()[
            i * num_outputs : (i + 1) * num_outputs,
            i * num_inputs : (i + 1) * num_inputs,
        ].copy_(linear_layer.weight)
    return new_layer
