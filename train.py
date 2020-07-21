import argparse
import itertools

import torch
import torch.nn.functional as F
import torch.optim as optim

from prime_mult.data import make_data_loader
from prime_mult.models import named_model


def main():
    args = arg_parser().parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = named_model(args.model_name, args.num_bits)
    model.to(device)
    data = iter(make_data_loader(args.num_bits, args.batch_size))
    opt = optim.Adam(model.parameters(), lr=args.lr)

    for i in itertools.count():
        inputs, targets = next(data)
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)
        loss = F.binary_cross_entropy_with_logits(outputs, targets)
        acc = compute_accuracy(outputs, targets)
        opt.zero_grad()
        loss.backward()
        opt.step()

        print(f"step {i:06}: loss={loss.item()} acc={acc.item()}")

        if i and not i % args.save_interval:
            print("saving...")
            torch.save(model.state_dict(), f"model_{i:06}.pt")


def compute_accuracy(outputs, targets):
    bool_out = outputs > 0
    bool_targ = targets > 0.5
    return (bool_out == bool_targ).float().mean()


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-bits", help="number of bits in primes", type=int, default=32
    )
    parser.add_argument("--batch-size", help="SGD batch size", type=int, default=32)
    parser.add_argument("--lr", help="Adam learning rate", type=float, default=1e-3)
    parser.add_argument("--model-name", help="type of model", type=str, default="mlp")
    parser.add_argument(
        "--save-interval", help="iterations per save", type=int, default=1000
    )
    return parser


if __name__ == "__main__":
    main()
