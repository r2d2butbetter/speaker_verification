import os
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

# Add the project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_io import get_splits
from lstm.model import SpeakerLSTM
from lstm.data import ShortUtteranceDataset

def compute_d_vectors(model, dataset, device):
    """
    Pass exactly chunked dataset audio through model and compute embeddings.
    Returns:
        dict: speaker_id -> list of embeddings (numpy arrays)
    """
    model.eval()
    speaker_embs = defaultdict(list)
    
    with torch.no_grad():
        for i in range(len(dataset)):
            feats, label = dataset[i] # label is the speaker index
            # Add batch dimension
            feats = feats.unsqueeze(0).to(device)
            
            # Scale features
            mean = feats.mean(dim=1, keepdim=True)
            std = feats.std(dim=1, keepdim=True) + 1e-6
            feats = (feats - mean) / std
            
            if model.num_classes is not None:
                emb, _ = model(feats) # Unpack since model returns (emb, logits)
            else:
                emb = model(feats)
                
            # Reconstruct speaker ID from label idx (need to map back)
            # Actually, this requires the wav path. Let's just use the dataset wav list.
            wav_path = Path(dataset.wavs[i])
            speaker_id = wav_path.parent.name
            
            speaker_embs[speaker_id].append(emb.cpu().numpy().flatten())
            
    return speaker_embs

def run_evaluation():
    lists_dir = Path("data/lists")
    enroll_list_path = lists_dir / "test_enrollment_list.txt"
    
    enroll_wavs = []
    test_wavs = []
    
    if not enroll_list_path.exists():
        print("Missing evaluation files. Did you run the prep script?")
        return
        
    with open(enroll_list_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split('|')
            if len(parts) == 2:
                # speaker_id = parts[0]
                paths = parts[1].split(',')
                # First n paths for enrollment, rest for testing. Let's use 3 for enroll, rest for test.
                # Standard TIMIT split per speaker
                if len(paths) >= 3:
                    enroll_wavs.extend(paths[:3])
                    test_wavs.extend(paths[3:])
                else:
                    enroll_wavs.extend(paths)
                    
    if not enroll_wavs or not test_wavs:
        print("Missing evaluation files. Did you run the prep script?")
        return
        
    print(f"Loaded {len(enroll_wavs)} enrollment files and {len(test_wavs)} test files.")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load Model
    model_path = Path("results/models/lstm/lstm_final.pt")
    if not model_path.exists():
        print(f"Model not found at {model_path}. Please train first.")
        return
        
    num_classes = 462 # Typical number of train speakers in TIMIT if correctly prep'd, 
                      # but for evaluation, the classifier layer isn't strictly needed for infer.
                      # Ideally we load the `num_classes` from the saved dict, but if we saved 
                      # state_dict only, we load dummy num_classes and ignore classifier.
                      
    checkpoint = torch.load(model_path, map_location=device)
    
    # Check if checkpoint is a dict with model state
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
        
    model = SpeakerLSTM(
        input_dim=39,
        hidden_dim=256,
        num_layers=2,
        embedding_dim=128,
        num_classes=None # Since we use TripletLoss
    )
    
    # Use strict=False to ignore any hanging classifier layers from previous training
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    # 2. Extract Enroll Embeddings
    print("Extracting enrollment d-vectors...")
    # Just passing dummy speaker2idx since we use paths to get speaker id in the function  
    dummy_dict = defaultdict(lambda: 0)
    enroll_dataset = ShortUtteranceDataset(
        list_of_wavs=enroll_wavs, speaker2idx=dummy_dict, chunk_length_sec=1.5, is_train=False
    )
    enroll_embs = compute_d_vectors(model, enroll_dataset, device)
    
    # Compute mean d-vector per speaker
    target_models = {}
    for spk, embs in enroll_embs.items():
        mean_emb = np.mean(embs, axis=0) # Average over utterances
        # L2 Normalize
        target_models[spk] = mean_emb / np.linalg.norm(mean_emb)
        
    print(f"Enrolled {len(target_models)} target speakers.")

    # 3. Extract Test Embeddings
    print("Extracting test d-vectors and scoring...")
    test_dataset = ShortUtteranceDataset(
        list_of_wavs=test_wavs, speaker2idx=dummy_dict, chunk_length_sec=1.5, is_train=False
    )
    
    test_embs = compute_d_vectors(model, test_dataset, device)
    
    # 4. Score all vs all (or positive vs negative)
    y_true = []
    y_scores = []
    
    # We will score every test utterance against every enrolled speaker
    # Target = True if spk == test test_spk
    for test_spk, t_embs in test_embs.items():
        for t_emb in t_embs:
            # L2 Normalize the test embedding
            t_emb_norm = t_emb / np.linalg.norm(t_emb)
            
            for enr_spk, enr_mean_emb in target_models.items():
                score = np.dot(t_emb_norm, enr_mean_emb) # Cosine similarity since normalized
                y_scores.append(score)
                y_true.append(1 if test_spk == enr_spk else 0)
                
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    # Evaluate Separation (e.g. EER)
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    
    # The EER is the point where FPR == FNR
    eer_threshold_idx = np.nanargmin(np.absolute((fnr - fpr)))
    eer = fpr[eer_threshold_idx]
    
    print(f"\n======================================")
    print(f"Evaluated on {len(y_scores)} trials.")
    print(f"Equal Error Rate (EER): {eer*100:.2f}%")
    print(f"======================================")
    
if __name__ == "__main__":
    run_evaluation()
