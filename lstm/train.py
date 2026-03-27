import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

# Add the project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_io import get_splits
from lstm.model import SpeakerLSTM
from lstm.data import ShortUtteranceDataset


def train_lstm():
    lists_dir = Path("data/lists")
    ubm_list_path = lists_dir / "ubm_train_list.txt"
    
    train_wavs = []
    if ubm_list_path.exists():
        with open(ubm_list_path, 'r') as f:
            train_wavs = [line.strip() for line in f if line.strip()]
    
    if not train_wavs:
        print("No background wavs found! Please run scripts/01_prep_timit.py first.")
        return

    print(f"Found {len(train_wavs)} background WAV files.")

    # 1. Gather all unique speaker IDs from the background set 
    # Assumes TIMIT path structure: .../TRAIN/DR1/FCJF0/SA1.WAV -> FCJF0
    speakers = set([Path(w).parent.name for w in train_wavs])
    speaker2idx = {spk: i for i, spk in enumerate(sorted(speakers))}
    num_classes = len(speakers)
    print(f"Training on {num_classes} background speakers.")

    # 2. Setup Dataset and DataLoader
    # We use 1.5 second chunks for short-duration training
    train_dataset = ShortUtteranceDataset(
        list_of_wavs=train_wavs,
        speaker2idx=speaker2idx,
        chunk_length_sec=1.5,
        is_train=True
    )
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)

    # 3. Initialize Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = SpeakerLSTM(
        input_dim=39,      # MFCC + delta + delta-delta (13 x 3)
        hidden_dim=256,
        num_layers=2,
        embedding_dim=128, # Size of the d-vector
        num_classes=None   # Remove classification layer for Triplet Loss
    ).to(device)

    # 4. Setup Optimizer and Loss Function
    criterion = nn.TripletMarginLoss(margin=1.0, p=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)

    # 5. Training Loop
    num_epochs = 25
    
    # Create results directory
    models_dir = Path("results/models/lstm")
    models_dir.mkdir(parents=True, exist_ok=True)

    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (anchor_feats, pos_feats, neg_feats) in enumerate(train_loader):
            def scale_feats(f):
                mean = f.mean(dim=1, keepdim=True)
                std = f.std(dim=1, keepdim=True) + 1e-6
                return (f - mean) / std
                
            anchor_feats = scale_feats(anchor_feats).to(device)
            pos_feats = scale_feats(pos_feats).to(device)
            neg_feats = scale_feats(neg_feats).to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            anchor_emb = model(anchor_feats)
            pos_emb = model(pos_feats)
            neg_emb = model(neg_feats)
            
            loss = criterion(anchor_emb, pos_emb, neg_emb)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients in LSTM
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            
        epoch_loss = total_loss / len(train_loader)
        # We don't have accuracy in triplet margin setup without hard negative mining computation overhead
        print(f"==> Epoch {epoch+1} Summary: Avg Triplet Loss: {epoch_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
        
        scheduler.step()
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
            'speaker1idx': speaker2idx # Save for reference if needed
        }, models_dir / f"lstm_epoch_{epoch+1}.pt")
        
    print("Training finished. Final model saved.")
    torch.save(model.state_dict(), models_dir / "lstm_final.pt")

if __name__ == "__main__":
    train_lstm()
