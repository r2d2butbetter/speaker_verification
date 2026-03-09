"""Parse TIMIT and create background/target lists.

This script is a placeholder scaffold. Fill paths and logic as needed.
"""

from pathlib import Path
import argparse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", type=Path, required=False, help="Path to raw TIMIT (SPHERE)")
    ap.add_argument("--wav-root", type=Path, required=False, help="Path to converted WAVs")
    ap.add_argument("--out", type=Path, default=Path("short_duration_sv/data/lists"), help="Output lists directory")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for name in ["background.txt", "enroll.txt", "test.txt"]:
        (args.out / name).write_text("# TODO: populate with IDs or paths\n", encoding="utf-8")

    print(f"Wrote placeholder lists to {args.out}")


if __name__ == "__main__":
    main()
