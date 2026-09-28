"""Independent paired-accuracy baseline; imports no TrialLedger code."""
import argparse
import csv
import json
import random
from pathlib import Path


def percentile(values, q):
    values = sorted(values)
    index = (len(values) - 1) * q
    low = int(index)
    return values[low] + (values[min(low + 1, len(values) - 1)] - values[low]) * (index - low)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--iterations', type=int, default=200)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    with args.input.open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    left = [int((float(r['a']) >= .5) == bool(int(r['y_true']))) for r in rows]
    right = [int((float(r['b']) >= .5) == bool(int(r['y_true']))) for r in rows]
    n = len(rows)
    a, b = sum(left) / n, sum(right) / n
    rng = random.Random(args.seed)
    boot = []
    for _ in range(args.iterations):
        indices = [rng.randrange(n) for _ in range(n)]
        boot.append(sum(left[i] for i in indices)/n - sum(right[i] for i in indices)/n)
    permutations = []
    for _ in range(args.iterations):
        a_sum = b_sum = 0
        for x,y in zip(left,right):
            if rng.getrandbits(1): x,y = y,x
            a_sum += x
            b_sum += y
        permutations.append(a_sum/n - b_sum/n)
    report = {'accuracy_a': a, 'accuracy_b': b, 'delta': a-b,
              'ci_low': percentile(boot,.025), 'ci_high': percentile(boot,.975),
              'p': (1+sum(abs(x)>=abs(a-b) for x in permutations))/(args.iterations+1)}
    args.output.write_text(json.dumps(report),encoding='utf-8')


if __name__ == '__main__':
    main()
