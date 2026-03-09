"""Extract background features and fit the generic GMM-UBM (placeholder)."""

from pathlib import Path
import argparse
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lists", type=Path, default=Path("short_duration_sv/data/lists"))
    ap.add_argument("--out", type=Path, default=Path("short_duration_sv/results/models/ubm.pkl"))
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Placeholder: generate dummy data to fit a tiny GMM if desired later
    np.save(args.out.with_suffix(".placeholder.npy"), np.zeros((1, 1), dtype=np.float32))
    print(f"Prepared placeholder output at {args.out}")


if __name__ == "__main__":
    main()
