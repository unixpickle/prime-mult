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
        x.backward(retain_graph=True)
        if inputs.grad is not None:
            layer.weight.detach()[i] = inputs.grad.float()
        else:
            layer.weight.detach()[i].zero_()
        if bias.grad is not None:
            layer.bias.detach()[i] = bias.grad.float()
        else:
            layer.bias.detach()[i] = 0
    return layer


def repeat_block_diagonal(linear_layer, repeats, sparse=False):
    if sparse:
        return BlockSparseDiagonal(linear_layer, repeats)
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


class BlockSparseDiagonal(nn.Module):
    """
    A block-sparse diagonal layer that is initialized by repeating a linear
    layer over the diagonal.
    """

    def __init__(self, linear_layer, num_blocks):
        super().__init__()
        self.num_blocks = num_blocks
        batch_bcast = torch.zeros(num_blocks, 1, 1).to(linear_layer.weight)
        self.weight = nn.Parameter(
            linear_layer.weight.detach() + batch_bcast, requires_grad=True
        )
        self.bias = nn.Parameter(
            linear_layer.bias.detach() + batch_bcast[:, 0], requires_grad=True
        )

    def forward(self, x):
        batch = x.shape[0]
        x = x.view(batch, self.num_blocks, -1)
        x = x.permute(1, 0, 2)
        x = torch.bmm(x, self.weight.permute(0, 2, 1))
        x = x.permute(1, 0, 2)
        x = x + self.bias
        x = x.reshape(batch, -1)
        return x
