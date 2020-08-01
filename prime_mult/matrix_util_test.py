import pytest

import torch
import torch.nn as nn

from .matrix_util import create_linear_layer, repeat_block_diagonal


def test_create_linear_layer():
    def my_function(inputs, bias):
        return (inputs[0] + 0.3 * inputs[1] + 0.5 * inputs[3], inputs[1] + bias * 2)

    lin_layer = create_linear_layer(my_function, 5)
    test_inputs = torch.randn(3, 5)
    actual = lin_layer(test_inputs)
    expected = torch.stack(
        [torch.stack(my_function(row, 1), dim=0) for row in test_inputs], dim=0
    )
    assert torch.abs(actual - expected).max().item() < 1e-4


@pytest.mark.parametrize("sparse", [False, True])
def test_repeat_block_diagonal(sparse):
    layer = nn.Linear(3, 5)
    rep_layer = repeat_block_diagonal(layer, 4, sparse=sparse)
    inputs = torch.randn(7, 12)
    actual = rep_layer(inputs)
    expected = torch.cat(
        [
            layer(inputs[:, :3]),
            layer(inputs[:, 3:6]),
            layer(inputs[:, 6:9]),
            layer(inputs[:, 9:12]),
        ],
        dim=-1,
    )
    assert torch.abs(actual - expected).max().item() < 1e-4
