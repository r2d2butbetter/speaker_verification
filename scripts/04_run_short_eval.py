"""Main experiment loop for short-duration evaluation (0.5s, 1.0s, 1.5s)."""

from pathlib import Path
import argparse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=Path, default=Path("short_duration_sv/results/models"))
    ap.add_argument("--scores", type=Path, default=Path("short_duration_sv/results/scores"))
    ap.add_argument("--plots", type=Path, default=Path("short_duration_sv/results/plots"))
    args = ap.parse_args()

    for p in [args.scores, args.plots]:
        p.mkdir(parents=True, exist_ok=True)
    print("Evaluation placeholder. Populate with scoring and plotting logic.")


if __name__ == "__main__":
    main()
