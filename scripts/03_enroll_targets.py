import copy
from pathlib import Path
import argparse
import pickle as pkl
import joblib

import librosa
import numpy as np
import sys

path_to_src = Path(__file__).parent.parent
path_to_src = str(path_to_src)
sys.path.append(path_to_src)

from src.features import extract_mfcc_features
from src.gmm_ubm import UBMModel, map_adapt

TIMIT_ROOT = Path("data") / "raw_timit" / "data"
LISTS_DIR = Path("data") / "lists"
OUT_DIR = Path("results")

def enroll_speakers():
    model_path = OUT_DIR / "models" / "ubm_model.pkl"


    ubm_gmm = joblib.load(model_path)
    ubm = UBMModel(gmm=ubm_gmm)
    print(f"Loaded ubm with {ubm.gmm.n_components} components")

    list_path = LISTS_DIR / "test_enrollment_list.txt"
    with open(list_path, 'r')as f:  
        lines = f.read().splitlines()
    
    print(f"enrolling {len(lines)} speakers")
    for line in lines:
        speaker_id, paths_str = line.split('|')
        all_paths = paths_str.split(',')

        enroll_paths = [p for p in all_paths if "SX" in Path(p).name]

        speaker_features = []
        for wav_path in enroll_paths:
            audio, sr = librosa.load(wav_path, sr=16000)
            
            clean_audio, _ = librosa.effects.trim(audio, top_db=25)
            features = extract_mfcc_features(clean_audio, sr)
            speaker_features.append(features)
    
        x_enroll = np.vstack(speaker_features)

        target_model = map_adapt(ubm, x_enroll, relevance_factor=16.0)

        clean_id = Path(speaker_id).name
        speaker_model_path = OUT_DIR / "enrolled_models"
        model_file_path = speaker_model_path / f"speaker_{clean_id}.pkl"
        speaker_model_path.mkdir(exist_ok=True, parents=True)

        with open(model_file_path, 'wb') as f:
            pkl.dump(target_model.gmm, f)

        print(f"Enrolled Speaker: {clean_id} | Frames: {x_enroll.shape[0]}")

    print("\nAll target speakers successfully enrolled and saved to disk!")




def main() -> None:
    enroll_speakers()


if __name__ == "__main__":
    main()
