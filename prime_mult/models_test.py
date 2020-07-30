import torch

from .models import PreInitMLP


def test_pre_init_mlp_4bits():
    mlp = PreInitMLP(num_bits=4)
    inputs = torch.tensor(
        [
            [1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0],  # 5 * 7
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # 15 * 15
        ]
    )
    expected = (
        torch.tensor(
            [
                [1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            ]
        )
        > 0.5
    )
    actual = mlp(inputs) > 0
    for a, x in zip(actual, expected):
        assert (a == x).all()
