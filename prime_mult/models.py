import torch
import torch.nn as nn


def named_model(name, num_bits):
    if name == "mlp":
        return MLPFactorizer(num_bits)
    elif name == "siren":
        return SIRENFactorizer(num_bits)
    raise ValueError(f"no such model: {name}")


class MLPFactorizer(nn.Module):
    def __init__(self, num_bits, d=1024):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.Linear(num_bits * 2, d),
            nn.GELU(),
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, num_bits * 2),
        )

    def forward(self, x):
        return self.sequential(x)


class SIRENFactorizer(nn.Module):
    def __init__(self, num_bits, d=1024):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.Linear(num_bits * 2, d),
            Sin(),
            nn.Linear(d, d),
            Sin(),
            nn.Linear(d, d),
            Sin(),
            nn.Linear(d, num_bits * 2),
        )

    def forward(self, x):
        return self.sequential(x)


class Sin(nn.Module):
    def forward(self, x):
        return torch.sin(x)
