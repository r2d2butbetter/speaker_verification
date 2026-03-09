"""MAP adapts the UBM for specific test speakers (placeholder)."""

from pathlib import Path
import argparse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=Path, default=Path("short_duration_sv/results/models"))
    ap.add_argument("--lists", type=Path, default=Path("short_duration_sv/data/lists"))
    args = ap.parse_args()

    args.models.mkdir(parents=True, exist_ok=True)
    print(f"Enrollment placeholder. Models directory: {args.models}")


if __name__ == "__main__":
    main()
