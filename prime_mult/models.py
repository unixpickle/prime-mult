import torch
import torch.nn as nn


def named_model(name, num_bits):
    if name == "mlp":
        return MLPFactorizer(num_bits)
    elif name == "siren":
        return SIRENFactorizer(num_bits)
    elif name == "gated":
        return GatedFactorizer(num_bits)
    elif name == "hardcoded":
        return HardCodedFactorizer(num_bits)
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


class GatedFactorizer(nn.Module):
    def __init__(self, num_bits, d=1024):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.Linear(num_bits * 2, d * 2),
            GatedAct(),
            nn.Linear(d, d * 2),
            GatedAct(),
            nn.Linear(d, d * 2),
            GatedAct(),
            nn.Linear(d, num_bits * 2),
        )

    def forward(self, x):
        return self.sequential(x)


class GatedAct(nn.Module):
    def forward(self, x):
        size = x.shape[1] // 2
        return x[:, :size] * torch.sigmoid(x[:, size:])


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


class HardCodedFactorizer(nn.Module):
    def __init__(self, num_bits):
        super().__init__()
        self.num_bits = num_bits
        self.logit_scale = nn.Parameter(torch.tensor([1.0]))

    def forward(self, x):
        results = []
        for elem in x:
            results.append(self._multiply(elem))
        return torch.stack(results) * self.logit_scale

    def _multiply(self, pq_pair):
        values = (pq_pair > 0.5).detach().cpu().numpy().tolist()
        p = _bits_to_number(values[: self.num_bits])
        q = _bits_to_number(values[self.num_bits :])
        pq = p * q
        return _number_to_tensor(self.num_bits * 2, pq).to(pq_pair)


def _bits_to_number(bits):
    result = 0
    for i, b in enumerate(bits):
        if b:
            result |= 1 << i
    return result


def _number_to_tensor(num_bits, n):
    return torch.tensor([(1.0 if (n & (1 << i)) else -1.0) for i in range(num_bits)])
