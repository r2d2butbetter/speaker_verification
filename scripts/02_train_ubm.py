"""Extract background features and fit the generic GMM-UBM (placeholder)."""

from pathlib import Path
import argparse
import sys
from xml.sax.handler import all_features
import librosa
import numpy as np
import pickle
from sklearn.mixture import GaussianMixture


path_to_src = Path(__file__).parent.parent
path_to_src = str(path_to_src)
sys.path.append(path_to_src)

from src.features import extract_mfcc_features

TIMIT_ROOT = Path("data") / "raw_timit" / "data"
LISTS_DIR = Path("data") / "lists"
OUT_DIR = Path("results")

def train_ubm_model():
    list_path = LISTS_DIR / "ubm_train_list.txt"
    model_dir = OUT_DIR / "models"
    model_dir.mkdir(exist_ok=True, parents=True)

    all_features =[]

    with open(list_path, 'r') as f:
        file_paths = f.read().splitlines()
    
    for i, wav_path in enumerate(file_paths):
        audio, sr = librosa.load(wav_path, sr=16000)

        clean_audio, _ = librosa.effects.trim(audio, top_db=25) # removing silence using energy
        features = extract_mfcc_features(clean_audio, sr)

        all_features.append(features)

        if(i%100==0): print(f"Processed {i+1} files")

    X_train = np.vstack(all_features)
    print(f"\nTotal acoustic frames extracted: {X_train.shape[0]}")
    print(f"Feature vector dimensionality: {X_train.shape[1]}")

    print("Training:")
    ubm = GaussianMixture(n_components=64, covariance_type='diag', max_iter=100, verbose=2)
    ubm.fit(X_train)

    model_path = model_dir / "ubm_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(ubm, f)



def main() -> None:
    train_ubm_model()


if __name__ == "__main__":
    main()
