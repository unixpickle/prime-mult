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


class PreInitMLP(nn.Module):
    def __init__(self, num_bits):
        super().__init__()

        log_bits = 1
        while 2 ** log_bits < num_bits:
            log_bits += 1
        if 2 ** log_bits != num_bits:
            raise ValueError(f"num_bits is {num_bits} but require a power of 2")

        self.num_bits = num_bits
        layers = [self._create_first_layer()]
        for i in range(log_bits):
            layers.append(self.addition_layer(i))
        self.network = nn.Sequential(*layers)

    def create_first_layer(self):
        """
        Create a layer that produces the rows for long addition.
        Each row is self.num_bits * 2 bits, but half of those bits will be 0s.

        If the input is <p, q>, the output is <p*q_0, p*q_1 << 1, p*q_2 << 2, ...>.
        """
        layer = nn.Linear(self.num_bits * 2, 2 * self.num_bits ** 2)
        w = layer.weight.detach()
        b = layer.bias.detach()
        w.zero_()
        b.zero_()
        for i in range(self.num_bits):
            for j in range(self.num_bits):
                # i is index in q, j is index in p.
                out_idx = i * (1 + self.num_bits * 2) + j
                w[out_idx, j] = 8.0
                w[out_idx, i + self.num_bits] = 8.0
                b[out_idx] = -12.0
        return nn.Sequential(layer, nn.Sigmoid())

    def addition_layer(self, depth):
        """
        Create a layer that adds pairs of two binary numbers together, which
        can be applied recursively to add many numbers.
        """
        # TODO: create carry bit layer
        # TODO: create xor and carry flags
        # TODO: compute xor between those flags to compute final sum

    def create_carry_bit_layer(self, depth):
        """
        Compute carry and propagate information for carry-lookahead addition.

        The input will be of the form <a1, b1, a2, b2, ...>, where each number
        is self.num_bits * 2 bits, and we want to compute <a1 + b1, ...>.

        The output will be of the form <*inputs, c1, p1, k1, c2, p2, k2, ...>,
        where cN, pN, and kN are carry, propagate, and kill bits.
        """
        num_pairs = self.num_bits / (2 ** (depth + 1))
        in_dims = self.num_bits * 4 * num_pairs
        layer = nn.Linear(in_dims, in_dims + 3 * num_pairs * self.num_bits * 2)
        w = layer.weight.detach()
        b = layer.bias.detach()
        w.zero_()
        b.zero_()

        # Identity for inputs.
        for i in range(in_dims):
            w[in_dims, in_dims] = 1.0

        for i in range(num_pairs):
            for j in range(self.num_bits * 2):
                a_idx = self.num_bits * 4 * i + j
                b_idx = a_idx + self.num_bits * 2
                carry_idx = in_dims + self.num_bits * 6 * i + j
                prop_idx = carry_idx + self.num_bits * 2
                kill_idx = prop_idx + self.num_bits * 2
                w[carry_idx, a_idx] = 8.0
                w[carry_idx, b_idx] = 8.0
                b[carry_idx] = -12.0
                w[prop_idx, a_idx] = 8.0
                w[prop_idx, b_idx] = 8.0
                b[prop_idx] = -4.0
                w[kill_idx, a_idx] = -8.0
                w[kill_idx, b_idx] = -8.0
                b[prop_idx] = 4.0

        return layer

    def create_xor_and_carry(self, depth):
        """
        Compute more intermediate values for computing a sum.

        The inputs are <a1, b1, ..., aN, bN, c1, p1, k1, ..., cN, pN, kN>.
        The outputs are <a1^b1, carry1, a2^b2, carry2, ...>.
        """
        num_pairs = self.num_bits / (2 ** (depth + 1))
        digit_dims = self.num_bits * 4 * num_pairs
        in_dims = digit_dims + 3 * num_pairs * self.num_bits * 2
        out_dims = num_pairs * self.num_bits * 2
        layer = nn.Linear(in_dims, out_dims)
        w = layer.weight.detach()
        b = layer.bias.detach()
        w.zero_()
        b.zero_()

        for i in range(num_pairs):
            for j in range(self.num_bits * 2):
                a_idx = self.num_bits * 4 * i + j
                b_idx = a_idx + self.num_bits * 2
                carry_idx = digit_dims + self.num_bits * 6 * i + j
                prop_idx = carry_idx + self.num_bits * 2
                kill_idx = prop_idx + self.num_bits * 2

                # Going from least bit to greatest bit, what we do is:
                #  - Compute (a ^ b), which is NOT(carry & kill) (just a sum).
                #  - XOR by current carry flag:
                #    - For a previous carry flag, we carry if there are no kills
                #      after the carry flag.

                # TODO: this.
