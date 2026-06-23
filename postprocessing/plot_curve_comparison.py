from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a compact comparison plot from CSV data.")
    parser.add_argument("csv", type=Path)
    parser.add_argument("--x-col", required=True)
    parser.add_argument("--num-col", required=True)
    parser.add_argument("--ref-col", required=True)
    parser.add_argument("--output", type=Path, default=Path("curve_comparison.pdf"))
    parser.add_argument("--ylabel", default="")
    args = parser.parse_args()

    data = np.genfromtxt(args.csv, delimiter=",", names=True)
    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    ax.plot(data[args.x_col], data[args.ref_col], "o", ms=3, label="reference")
    ax.plot(data[args.x_col], data[args.num_col], "-", lw=1.5, label="simulation")
    ax.set_xlabel(args.x_col)
    ax.set_ylabel(args.ylabel or args.num_col)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.output)


if __name__ == "__main__":
    main()

