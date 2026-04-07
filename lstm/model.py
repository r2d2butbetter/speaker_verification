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
        
        # Self-Attention layer to dynamically focus on informative frames
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False)
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
        # Use attention to compute a weighted sum over the time dimension
        attn_weights = F.softmax(self.attention(lstm_out), dim=1) # (batch, time, 1)
        pooled = torch.sum(lstm_out * attn_weights, dim=1) # (batch, hidden_dim * 2)
        
        # Project to embedding space with an MLP head for better metric learning
        x_proj = F.relu(self.linear(pooled))
        
        # L2 Normalize the embedding (crucial for cosine similarity and triplet margin)
        embedding = F.normalize(x_proj, p=2, dim=1)
        
        if self.num_classes is not None:
            logits = self.classifier(embedding)
            return embedding, logits
            
        return embedding
