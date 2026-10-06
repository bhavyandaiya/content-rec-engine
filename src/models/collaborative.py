"""Neural Collaborative Filtering / Two-Tower Architecture in PyTorch.

Maps reader IDs and article IDs into a joint latent embedding space.
Optimizes prediction of composite implicit telemetry affinity scores.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CollaborativeDataset(Dataset):
    """PyTorch Dataset for user-item implicit affinity interactions."""

    def __init__(
        self,
        user_indices: np.ndarray,
        item_indices: np.ndarray,
        affinity_targets: np.ndarray
    ):
        self.user_indices = torch.tensor(user_indices, dtype=torch.long)
        self.item_indices = torch.tensor(item_indices, dtype=torch.long)
        self.affinity_targets = torch.tensor(affinity_targets, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.user_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.user_indices[idx], self.item_indices[idx], self.affinity_targets[idx]


class UserTower(nn.Module):
    """User tower projecting discrete user ID to latent dense vector space."""

    def __init__(self, num_users: int, embedding_dim: int = 64, hidden_dims: List[int] = [128, 64], dropout: float = 0.1):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)

        layers = []
        prev_dim = embedding_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, embedding_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, user_idx: torch.Tensor) -> torch.Tensor:
        raw_emb = self.user_embedding(user_idx)
        proj = self.mlp(raw_emb)
        return F.normalize(proj, p=2, dim=-1)


class ItemTower(nn.Module):
    """Item tower projecting discrete article ID to latent dense vector space."""

    def __init__(self, num_items: int, embedding_dim: int = 64, hidden_dims: List[int] = [128, 64], dropout: float = 0.1):
        super().__init__()
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        layers = []
        prev_dim = embedding_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, embedding_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, item_idx: torch.Tensor) -> torch.Tensor:
        raw_emb = self.item_embedding(item_idx)
        proj = self.mlp(raw_emb)
        return F.normalize(proj, p=2, dim=-1)


class TwoTowerRecommender(nn.Module):
    """Two-Tower Collaborative Filtering Model combining User and Item latent representations."""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]
            
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim

        self.user_tower = UserTower(num_users, embedding_dim=embedding_dim, hidden_dims=hidden_dims, dropout=dropout)
        self.item_tower = ItemTower(num_items, embedding_dim=embedding_dim, hidden_dims=hidden_dims, dropout=dropout)

        # Mapping dictionaries from raw IDs to index and vice-versa
        self.user2idx: Dict[str, int] = {}
        self.idx2user: Dict[int, str] = {}
        self.item2idx: Dict[str, int] = {}
        self.idx2item: Dict[int, str] = {}

    def forward(self, user_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
        """Forward pass computing normalized dot-product similarity mapped to [0, 1]."""
        u_emb = self.user_tower(user_idx)  # (batch_size, embedding_dim)
        i_emb = self.item_tower(item_idx)  # (batch_size, embedding_dim)
        
        # Dot product of normalized vectors yields cosine similarity [-1.0, 1.0]
        cos_sim = torch.sum(u_emb * i_emb, dim=-1)
        # Rescale cosine similarity from [-1, 1] to [0, 1] for continuous affinity prediction
        pred_affinity = (cos_sim + 1.0) / 2.0
        return pred_affinity

    def get_user_vector(self, user_idx: int) -> np.ndarray:
        """Extract latent representation for single user index."""
        self.eval()
        with torch.no_grad():
            tensor_idx = torch.tensor([user_idx], dtype=torch.long, device=next(self.parameters()).device)
            vec = self.user_tower(tensor_idx).squeeze(0).cpu().numpy()
        return vec

    def get_item_vector(self, item_idx: int) -> np.ndarray:
        """Extract latent representation for single item index."""
        self.eval()
        with torch.no_grad():
            tensor_idx = torch.tensor([item_idx], dtype=torch.long, device=next(self.parameters()).device)
            vec = self.item_tower(tensor_idx).squeeze(0).cpu().numpy()
        return vec

    def get_all_item_embeddings(self) -> np.ndarray:
        """Extract all article embeddings as matrix (num_items, embedding_dim)."""
        self.eval()
        with torch.no_grad():
            all_indices = torch.arange(self.num_items, dtype=torch.long, device=next(self.parameters()).device)
            embeddings = self.item_tower(all_indices).cpu().numpy()
        return embeddings

    def predict_all_for_user(self, user_idx: int) -> np.ndarray:
        """Compute CF affinity predictions across all catalog items for given user."""
        u_vec = self.get_user_vector(user_idx)  # (embedding_dim,)
        all_items = self.get_all_item_embeddings()  # (num_items, embedding_dim)
        cos_sim = np.dot(all_items, u_vec)  # (num_items,)
        scores = (cos_sim + 1.0) / 2.0
        return np.clip(scores, 0.0, 1.0)

    def set_id_mappings(self, user2idx: Dict[str, int], item2idx: Dict[str, int]) -> None:
        """Assign ID string translation mappings."""
        self.user2idx = user2idx
        self.idx2user = {v: k for k, v in user2idx.items()}
        self.item2idx = item2idx
        self.idx2item = {v: k for k, v in item2idx.items()}

    def save_checkpoint(self, filepath: str) -> None:
        """Save model weights, architecture parameters, and vocabulary dictionaries."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        checkpoint = {
            "num_users": self.num_users,
            "num_items": self.num_items,
            "embedding_dim": self.embedding_dim,
            "state_dict": self.state_dict(),
            "user2idx": self.user2idx,
            "item2idx": self.item2idx
        }
        torch.save(checkpoint, filepath)
        logger.info(f"Model checkpoint successfully saved to {filepath}")

    @classmethod
    def load_checkpoint(cls, filepath: str, map_location: str = "cpu") -> "TwoTowerRecommender":
        """Load model weights and dictionary maps from checkpoint file."""
        checkpoint = torch.load(filepath, map_location=map_location)
        model = cls(
            num_users=checkpoint["num_users"],
            num_items=checkpoint["num_items"],
            embedding_dim=checkpoint["embedding_dim"]
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.set_id_mappings(checkpoint["user2idx"], checkpoint["item2idx"])
        model.eval()
        logger.info(f"Model loaded from {filepath} (users: {model.num_users}, items: {model.num_items})")
        return model
