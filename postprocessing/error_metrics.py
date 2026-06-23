from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def relative_l2_error(y_num: np.ndarray, y_ref: np.ndarray) -> float:
    return float(np.linalg.norm(y_num - y_ref) / np.linalg.norm(y_ref))


def relative_peak_error(y_num: np.ndarray, y_ref: np.ndarray) -> float:
    return float(abs(np.max(y_num) - np.max(y_ref)) / abs(np.max(y_ref)))


def final_rise_error(t_num: np.ndarray, t_ref: np.ndarray, t0: float) -> float:
    rise_num = t_num[-1] - t0
    rise_ref = t_ref[-1] - t0
    return float(abs(rise_num - rise_ref) / abs(rise_ref))


def load_columns(path: Path, num_col: str, ref_col: str) -> tuple[np.ndarray, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True)
    return np.asarray(data[num_col], dtype=float), np.asarray(data[ref_col], dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute relative curve-comparison metrics.")
    parser.add_argument("csv", type=Path)
    parser.add_argument("--num-col", required=True)
    parser.add_argument("--ref-col", required=True)
    parser.add_argument("--initial-temperature", type=float)
    args = parser.parse_args()

    y_num, y_ref = load_columns(args.csv, args.num_col, args.ref_col)
    print(f"relative_L2 = {100.0 * relative_l2_error(y_num, y_ref):.4f}%")
    print(f"relative_peak = {100.0 * relative_peak_error(y_num, y_ref):.4f}%")
    if args.initial_temperature is not None:
        print(
            "final_rise_error = "
            f"{100.0 * final_rise_error(y_num, y_ref, args.initial_temperature):.4f}%"
        )


if __name__ == "__main__":
    main()

