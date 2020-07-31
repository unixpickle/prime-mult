import torch
import torch.nn as nn

from . import matrix_util


def named_model(name, num_bits):
    if name == "mlp":
        return MLPFactorizer(num_bits)
    elif name == "siren":
        return SIRENFactorizer(num_bits)
    elif name == "gated":
        return GatedFactorizer(num_bits)
    elif name == "hardcoded":
        return HardCodedFactorizer(num_bits)
    elif name == "preinit":
        return PreInitMLP(num_bits)
    elif name == "preinit-sparse":
        return PreInitMLP(num_bits, sparse=True)
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
    def __init__(self, num_bits, saturation=10.0, sparse=False):
        super().__init__()
        self.num_bits = num_bits
        self.saturation = saturation
        self.sparse = sparse

        log_bits = 1
        while 2 ** log_bits < num_bits:
            log_bits += 1
        if 2 ** log_bits != num_bits:
            raise ValueError(f"num_bits is {num_bits} but require a power of 2")

        self.num_bits = num_bits
        layers = [self.create_first_layer()]
        for i in range(log_bits):
            if i > 0:
                layers.append(nn.Sigmoid())
            layers.append(self.addition_layer(i))
        self.network = nn.Sequential(*layers)
        for param in self.parameters():
            param.detach().mul_(saturation)

    def forward(self, x):
        return self.network(x)

    def create_first_layer(self):
        """
        Create a layer that produces the rows for long addition.
        Each row is self.num_bits * 2 bits, but half of those bits will be 0s.

        If the input is <p, q>, the output is <p*q_0, p*q_1 << 1, p*q_2 << 2, ...>.
        """

        def mask_function(inputs, bias):
            p = inputs[: self.num_bits]
            q = inputs[self.num_bits :]
            results = []
            for i, q_i in enumerate(q):
                results.append(-bias * torch.ones(i).to(inputs))
                results.append((p + q_i) * 2.0 - bias * 3.0)
                results.append(-bias * torch.ones(self.num_bits - i).to(inputs))
            return torch.cat(results, dim=0)

        linear_layer = matrix_util.create_linear_layer(mask_function, self.num_bits * 2)
        return nn.Sequential(linear_layer, nn.Sigmoid())

    def addition_layer(self, depth):
        """
        Create a layer that adds pairs of two binary numbers together, which
        can be applied recursively to add many numbers.
        """
        layers = [self.create_carry_bit_layer(depth)]
        for i in range(self.num_bits * 2):
            layers.append(self.create_bit_adder_layer(depth, i))
        layers.append(self.create_bit_carry_xor_layer(depth))
        return nn.Sequential(*layers)

    def create_carry_bit_layer(self, depth):
        """
        Compute carry and propagate information for carry-lookahead addition.

        The input will be of the form <a1, b1, a2, b2, ...>, where each number
        is self.num_bits * 2 bits, and we want to compute <a1 + b1, ...>.

        The output will be of the form <c11, p11, k11, c12, p12, k12, ...>,
        where cNj, pNj, and kNj are carry, propagate, and kill bits.
        """

        def bit_function(inputs, bias):
            a = inputs[: self.num_bits * 2]
            b = inputs[self.num_bits * 2 :]
            carry = (a + b) * 2.0 - bias * 3.0
            prop = (a + b) * 2.0 - bias * 1.0
            kill = -prop
            return torch.stack([carry, prop, kill], dim=-1).view(-1)

        num_pairs = self.num_bits // (2 ** (depth + 1))
        linear_layer = matrix_util.create_linear_layer(bit_function, self.num_bits * 4)
        linear_layer = matrix_util.repeat_block_diagonal(
            linear_layer, num_pairs, sparse=self.sparse
        )
        return nn.Sequential(linear_layer, nn.Sigmoid())

    def create_bit_adder_layer(self, depth, bit_idx):
        """
        Create a layer that performs a 1-bit addition on the output of the
        carry bit layer.

        The input is of the form <c1, p1, k1, ...>.
        After each bit is computed, the k will be replaced by the XOR of the
        two bits, p will be replaced by a flag indicating if a carry from the
        previous bit has been propagated to the next bit.
        Thus, (p OR c) gives the true carry bit for the next bit.
        """

        def bit_function(inputs, bias):
            result = list(inputs * 2 - bias)

            c_cur = inputs[bit_idx * 3]
            p_cur = inputs[bit_idx * 3 + 1]
            k_cur = inputs[bit_idx * 3 + 2]

            if bit_idx == 0:
                next_carry = -1.0 * bias
            else:
                c_prev = inputs[(bit_idx - 1) * 3]
                p_prev = inputs[(bit_idx - 1) * 3 + 1]
                # (c_prev OR p_prev) AND p_cur
                next_carry = 2.0 * (c_prev + p_prev) - 5.0 * bias + 4.0 * p_cur
            result[bit_idx * 3 + 1] = next_carry

            xor = 1.0 * bias - 2.0 * (c_cur + k_cur)
            result[bit_idx * 3 + 2] = xor

            return torch.stack(result, dim=0)

        num_pairs = self.num_bits // (2 ** (depth + 1))
        linear_layer = matrix_util.create_linear_layer(bit_function, self.num_bits * 6)
        linear_layer = matrix_util.repeat_block_diagonal(
            linear_layer, num_pairs, sparse=self.sparse
        )
        return nn.Sequential(linear_layer, nn.Sigmoid())

    def create_bit_carry_xor_layer(self, depth):
        """
        Create a layer that turns the output of a chain of bit adders into a
        final sum between the digits and the carry bits.
        This performs an XOR, and as such is more than one linear layer.

        Turns <c1, p1, xor1, ...> into <sum1, sum2, ...>.
        """

        def create_00_and_11(inputs, bias):
            results = []
            for i in range(self.num_bits * 2):
                if i == 0:
                    carry_prev = -1.0
                else:
                    c_prev = inputs[(i - 1) * 3]
                    p_prev = inputs[(i - 1) * 3 + 1]
                    carry_prev = -1.0 * bias + 2.0 * (c_prev + p_prev)

                # carry_prev is either -1, 1, or 3.
                xor_cur = inputs[i * 3 + 2]  # either 0 or 1

                # compute NOT(carry_prev OR xor_cur)
                results.append(bias - 4.0 * xor_cur - 2.0 * carry_prev)
                # compute (carry_prev AND xor_cur)
                results.append(-7.0 * bias + 6.0 * xor_cur + 2.0 * carry_prev)
            return torch.stack(results, dim=0)

        def create_xor(inputs, bias):
            bits = list(inputs)
            results = []
            for i in range(self.num_bits * 2):
                case1, case2 = bits[i * 2], bits[i * 2 + 1]
                results.append(bias - 2.0 * (case1 + case2))
            return torch.stack(results, dim=0)

        num_pairs = self.num_bits // (2 ** (depth + 1))
        lin_1 = matrix_util.create_linear_layer(create_00_and_11, self.num_bits * 6)
        lin_1 = matrix_util.repeat_block_diagonal(lin_1, num_pairs, sparse=self.sparse)
        lin_2 = matrix_util.create_linear_layer(create_xor, self.num_bits * 4)
        lin_2 = matrix_util.repeat_block_diagonal(lin_2, num_pairs, sparse=self.sparse)
        return nn.Sequential(lin_1, nn.Sigmoid(), lin_2)
