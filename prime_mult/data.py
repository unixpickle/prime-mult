"""
Random synthetic data generation
"""

import Crypto.Random as Random
from Crypto.Util import number
import torch


def make_data_loader(num_bits, batch_size, num_workers=4):
    """
    Create a data loader for prime multiplication problems.
    """
    dataset = PrimeProductDataset(num_bits)
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers
    )


class PrimeProductDataset(torch.utils.data.IterableDataset):
    """
    A PyTorch dataset that produces primes and their products.

    Inputs are represented as two N bit numbers, and outputs are represented
    as a single 2*N bit number. Values are zero-padded as necessary.
    """

    def __init__(self, num_bits):
        super().__init__()
        self.num_bits = num_bits

    def __iter__(self):
        rng = Random.new()
        while True:
            p = number.getPrime(self.num_bits, randfunc=rng.read)
            q = number.getPrime(self.num_bits, randfunc=rng.read)
            if p >= 2 ** self.num_bits or q >= 2 ** self.num_bits:
                # Workaround a bug in the crypto API.
                continue
            if p > q:
                p, q = q, p
            pq = p * q
            inputs = binary_number(self.num_bits, p) + binary_number(self.num_bits, q)
            outputs = binary_number(self.num_bits * 2, pq)
            yield (
                torch.tensor(inputs, dtype=torch.float),
                torch.tensor(outputs, dtype=torch.float),
            )


def binary_number(num_bits, n):
    return [(1.0 if (n & (1 << i)) else 0.0) for i in range(num_bits)]
