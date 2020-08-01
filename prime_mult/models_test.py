import pytest

import torch

from .models import PreInitMLP


@pytest.mark.parametrize("sparse", [False, True])
def test_pre_init_mlp_4bits(sparse):
    mlp = PreInitMLP(num_bits=4, sparse=sparse)
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
    mlp_out = mlp(inputs)
    actual = mlp_out > 0
    for a, x in zip(actual, expected):
        assert (a == x).all()
    # Outputs should be very saturated.
    assert not (mlp_out.abs() < 1.0).any()


def test_pre_init_mlp_create(benchmark):
    benchmark(lambda: PreInitMLP(8))
