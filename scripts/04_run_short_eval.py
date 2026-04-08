import pickle
import joblib
import numpy as np
import random
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from sklearn.metrics import roc_curve

# Point Python to your src/ directory
sys.path.append(str(Path(__file__).parent.parent))
from src.features import extract_mfcc_features
from src.dsp_utils import extract_longest_word

TIMIT_ROOT = Path("data") / "raw_timit" / "data"
LISTS_DIR = Path("data") / "lists"
OUT_DIR = Path("results")

def evaluate_short_duration_sv():
    list_path = LISTS_DIR / "test_enrollment_list.txt"
    model_dir = OUT_DIR
    
    # 1. Load the Universal Background Model
    ubm = joblib.load(model_dir / "models" / "ubm_model.pkl")
        
    # 2. Parse the test list and load all Target GMMs into RAM
    with open(list_path, 'r') as f:
        lines = f.read().splitlines()

    test_data = {}
    target_models = {}
    
    print("Loading Speaker Models into memory...")
    for line in lines:
        speaker_id, paths_str = line.split('|')
        clean_id = Path(speaker_id).name
        
        # Isolate the unseen 'SI' files for testing
        test_paths = [p for p in paths_str.split(',') if "SI" in Path(p).name]
        test_data[clean_id] = test_paths
        
        # Load this speaker's custom model
        model_path = model_dir / "enrolled_models" / f"speaker_{clean_id}.pkl"
        if model_path.exists():
            with open(model_path, 'rb') as f:
                target_models[clean_id] = pickle.load(f)

    speaker_ids = list(target_models.keys())
    
    true_scores = []
    impostor_scores = []

    print(f"\nRunning Wake-Word Evaluation for {len(speaker_ids)} enrolled speakers...")
    
    # 3. The Evaluation Loop
    for target_id in speaker_ids:
        target_gmm = target_models[target_id]
        test_files = test_data.get(target_id, [])
        
        for wav_path_str in test_files:
            wav_path = Path(wav_path_str)
            wrd_path = wav_path.with_suffix('').with_suffix('.WRD')
            
            try:
                # A. Extract the < 1.0s wake word
                audio, sr, word, duration = extract_longest_word(wav_path, wrd_path)
                
                # B. Convert to 60-D features
                # librosa deltas need at least 9 frames (~0.1s). If a word is too short, we skip.
                if len(audio) < int(sr * 0.1):
                    continue
                    
                features = extract_mfcc_features(audio, sr)
                
                # C. Calculate generic Background Score (Average log-likelihood per frame)
                score_ubm = ubm.score(features)
                
                # D. True Trial: Score against their OWN model
                score_true = target_gmm.score(features)
                llr_true = score_true - score_ubm
                true_scores.append(llr_true)
                
                # E. Impostor Trials: Score against 3 RANDOM OTHER models
                impostors = random.sample([s for s in speaker_ids if s != target_id], 3)
                for imp_id in impostors:
                    imp_gmm = target_models[imp_id]
                    score_imp = imp_gmm.score(features)
                    llr_imp = score_imp - score_ubm
                    impostor_scores.append(llr_imp)
                    
            except Exception as e:
                print(f"Skipping {wav_path.name} due to feature extraction error: {e}")

    print(f"\nEvaluation Complete!")
    print(f"Total True Trials computed:     {len(true_scores)}")
    print(f"Total Impostor Trials computed: {len(impostor_scores)}")
    
    eer, eer_threshold = calculate_eer(true_scores, impostor_scores)
    if eer is not None:
        print(f"Equal Error Rate (EER):         {eer * 100:.2f}%")
        print(f"EER Threshold (LLR):            {eer_threshold:.4f}")
    else:
        print("Equal Error Rate (EER):         unavailable (need both genuine and impostor scores)")

    plot_score_distributions(true_scores, impostor_scores, eer, eer_threshold)


def calculate_eer(true_scores, impostor_scores):
    if not true_scores or not impostor_scores:
        return None, None

    y_true = np.array([1] * len(true_scores) + [0] * len(impostor_scores))
    y_scores = np.array(true_scores + impostor_scores)

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fnr - fpr))
    return float(fpr[eer_idx]), float(thresholds[eer_idx])

def plot_score_distributions(true_scores, impostor_scores, eer=None, eer_threshold=None):
    all_scores = true_scores + impostor_scores
    score_mean = np.mean(all_scores)
    true_scores = [s - score_mean for s in true_scores]
    impostor_scores = [s - score_mean for s in impostor_scores]
    if eer_threshold is not None:
        eer_threshold -= score_mean

    plt.figure(figsize=(10, 6))
    
    plt.hist(impostor_scores, bins=40, density=True, alpha=0.6, color='red', label='Impostors (Spoofs)')
    plt.hist(true_scores, bins=40, density=True, alpha=0.6, color='green', label='True Speakers (Targets)')
    
    plt.title("Wake-Word Speaker Verification (< 0.8s Duration)\nLog-Likelihood Ratio Distributions")
    plt.xlabel("Log-Likelihood Ratio Score (LLR)")
    plt.ylabel("Probability Density")
    plt.axvline(x=0, color='black', linestyle='--', linewidth=1, label='Zero Threshold')
    if eer_threshold is not None:
        plt.axvline(x=eer_threshold, color='blue', linestyle=':', linewidth=1.5, label=f'EER Threshold ({eer_threshold:.3f})')
    if eer is not None:
        plt.text(0.02, 0.98, f"EER: {eer * 100:.2f}%", transform=plt.gca().transAxes, va='top')
    
    max_score = max(abs(np.min(all_scores - score_mean)), abs(np.max(all_scores - score_mean)))
    plt.xlim(-max_score, max_score)

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def main() -> None:
    evaluate_short_duration_sv()

if __name__ == "__main__":
    main()