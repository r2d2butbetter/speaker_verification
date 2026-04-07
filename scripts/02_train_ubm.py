"""Extract background features from TIMIT TRAIN and fit the GMM-UBM."""

import sys
from pathlib import Path
import argparse
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from src.data_io import load_wav
from src.features import mfcc_with_deltas
from src.gmm_ubm import train_ubm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lists", type=Path, default=Path("data/lists"))
    ap.add_argument("--out", type=Path, default=Path("results/models/ubm_model.pkl"))
    ap.add_argument("--n-components", type=int, default=64)
    ap.add_argument("--max-iter", type=int, default=100)
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Skip training if model already exists
    if args.out.exists():
        print(f"UBM already exists at {args.out}. Delete it to retrain.")
        return

    ubm_list = args.lists / "ubm_train_list.txt"
    if not ubm_list.exists():
        print("Missing ubm_train_list.txt. Run scripts/01_prep_timit.py first.")
        return

    with open(ubm_list) as f:
        wav_paths = [l.strip() for l in f if l.strip()]

    print(f"Extracting features from {len(wav_paths)} background files...")
    all_frames = []
    for i, wp in enumerate(wav_paths):
        try:
            audio, sr = load_wav(wp)
            feats = mfcc_with_deltas(np.array(audio), sr)  # (39, T)
            all_frames.append(feats.T)  # (T, 39)
        except Exception as e:
            print(f"  Skipping {wp}: {e}")
        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(wav_paths)}")

    X = np.vstack(all_frames).astype(np.float64)
    print(f"Total feature frames: {X.shape[0]} x {X.shape[1]}")

    print(f"Training UBM with {args.n_components} components...")
    ubm = train_ubm(X, n_components=args.n_components, max_iter=args.max_iter)
    ubm.save(args.out)
    print(f"UBM saved to {args.out}")


if __name__ == "__main__":
    main()
