"""MAP-adapt the UBM for each test speaker using their enrollment utterances."""

import sys
from pathlib import Path
import argparse
import numpy as np
import joblib

sys.path.append(str(Path(__file__).parent.parent))

from src.data_io import load_wav
from src.features import mfcc_with_deltas
from src.gmm_ubm import UBMModel, map_adapt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ubm", type=Path, default=Path("results/models/ubm_model.pkl"))
    ap.add_argument("--lists", type=Path, default=Path("data/lists"))
    ap.add_argument("--out", type=Path, default=Path("results/enrolled_models"))
    ap.add_argument("--n-enroll", type=int, default=3,
                    help="Number of utterances per speaker used for enrollment")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    if not args.ubm.exists():
        print(f"UBM not found at {args.ubm}. Train it first (scripts/02_train_ubm.py).")
        return

    ubm = UBMModel.load(args.ubm)
    print(f"Loaded UBM with {ubm.gmm.n_components} components.")

    enroll_list = args.lists / "test_enrollment_list.txt"
    if not enroll_list.exists():
        print("Missing test_enrollment_list.txt. Run scripts/01_prep_timit.py first.")
        return

    with open(enroll_list) as f:
        lines = [l.strip() for l in f if l.strip()]

    enrolled = 0
    for line in lines:
        parts = line.split("|")
        if len(parts) != 2:
            continue
        speaker_dir = parts[0]
        speaker_id = Path(speaker_dir).name
        wav_paths = parts[1].split(",")

        enroll_paths = wav_paths[:args.n_enroll]

        # Extract features from enrollment utterances
        frames = []
        for wp in enroll_paths:
            try:
                audio, sr = load_wav(wp)
                feats = mfcc_with_deltas(np.array(audio), sr)
                frames.append(feats.T)
            except Exception as e:
                print(f"  Skipping {wp}: {e}")

        if not frames:
            print(f"  No usable enrollment data for {speaker_id}, skipping.")
            continue

        X = np.vstack(frames).astype(np.float64)
        target = map_adapt(ubm, X)

        out_path = args.out / f"speaker_{speaker_id}.pkl"
        joblib.dump(target, out_path)
        enrolled += 1

    print(f"Enrolled {enrolled} target speakers in {args.out}")


if __name__ == "__main__":
    main()
