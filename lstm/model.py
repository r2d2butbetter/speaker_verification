"""LSTM Model for speaker verification short utterances."""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SpeakerLSTM(nn.Module):
    def __init__(self, input_dim=39, hidden_dim=256, num_layers=2, embedding_dim=128, num_classes=None):
        """
        Args:
            input_dim: Dimension of input features (default 39 for MFCC + deltas)
            hidden_dim: Hidden dimension of LSTM
            num_layers: Number of LSTM layers
            embedding_dim: Dimension of the extracted d-vector
            num_classes: Number of speakers for classification training (if not using contrastive)
        """
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        # We use bidirectional LSTM, so hidden size is doubled
        self.linear = nn.Linear(hidden_dim * 2, embedding_dim)
        
        self.num_classes = num_classes
        if num_classes is not None:
            self.classifier = nn.Linear(embedding_dim, num_classes)
            
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch, time, features)
        Returns:
            If num_classes is None:
                embeddings of shape (batch, embedding_dim)
            Else:
                (embeddings, logits)
        """
        # x is (batch, time, input_dim)
        lstm_out, (hn, cn) = self.lstm(x)
        
        # lstm_out is (batch, time, hidden_dim * 2)
        # We can take the mean over the time dimension (mean pooling)
        # or just take the last hidden state. Let's use mean pooling.
        pooled = torch.mean(lstm_out, dim=1)
        
        # Project to embedding space with an MLP head for better metric learning
        x_proj = F.relu(self.linear(pooled))
        
        # We can add another linear layer if desired, or just use the one
        # Here we'll just use self.linear as the first part of the MLP if we had two,
        # but let's just make sure the final embedding is L2 normalized.
        
        # L2 Normalize the embedding (crucial for cosine similarity and triplet margin)
        embedding = F.normalize(x_proj, p=2, dim=1)
        
        if self.num_classes is not None:
            logits = self.classifier(embedding)
            return embedding, logits
            
        return embedding
