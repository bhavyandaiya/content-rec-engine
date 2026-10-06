"""Content-Based Recommendation Engine.

Computes semantic affinity between user reading histories and candidate articles
using dense transformer embeddings and cosine similarity.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ContentBasedRecommender:
    """Ranks content by projecting reader interaction histories into semantic embedding space."""

    def __init__(self, item_embeddings: Optional[Dict[str, np.ndarray]] = None):
        """Initialize with dictionary mapping article_id to L2-normalized embedding vector."""
        self.item_embeddings: Dict[str, np.ndarray] = item_embeddings or {}
        self.item_ids: List[str] = list(self.item_embeddings.keys())
        self._embedding_matrix: Optional[np.ndarray] = None
        self._sync_matrix()

    def set_embeddings(self, embeddings_dict: Dict[str, np.ndarray]) -> None:
        """Update or register article embeddings."""
        self.item_embeddings = embeddings_dict
        self.item_ids = list(embeddings_dict.keys())
        self._sync_matrix()

    def _sync_matrix(self) -> None:
        """Construct fast contiguous matrix for batch dot-product scoring."""
        if self.item_embeddings:
            self._embedding_matrix = np.vstack([self.item_embeddings[aid] for aid in self.item_ids])
        else:
            self._embedding_matrix = None

    def build_user_vector_from_history(
        self,
        interactions: List[Dict[str, Any]],
        decay_factor: float = 0.95
    ) -> Optional[np.ndarray]:
        """Aggregate article embeddings weighted by composite affinity and recency decay.
        
        Formula:
            u_vec = sum_i(affinity_i * decay^i * v_i) / sum_i(affinity_i * decay^i)
        """
        valid_vectors = []
        weights = []

        for idx, event in enumerate(interactions):
            art_id = event.get("article_id")
            if art_id in self.item_embeddings:
                aff = float(event.get("affinity_score", 0.5))
                # Apply recency decay based on index order (assuming reverse-chronological)
                w = max(0.01, aff) * (decay_factor ** idx)
                valid_vectors.append(self.item_embeddings[art_id])
                weights.append(w)

        if not valid_vectors:
            return None

        mat = np.array(valid_vectors)  # (N, D)
        w_arr = np.array(weights).reshape(-1, 1)  # (N, 1)

        aggregated = np.sum(mat * w_arr, axis=0) / (np.sum(w_arr) + 1e-9)
        norm = np.linalg.norm(aggregated)
        if norm > 1e-9:
            aggregated = aggregated / norm

        return aggregated

    def score_all_items_for_user(self, user_vector: np.ndarray) -> Dict[str, float]:
        """Compute cosine similarity score across all catalog items for a given user vector."""
        if self._embedding_matrix is None or len(self.item_ids) == 0:
            return {}

        u_vec = user_vector.reshape(1, -1)
        sims = np.dot(u_vec, self._embedding_matrix.T).squeeze(0)  # Cosine similarity in [-1, 1]
        
        # Rescale [-1, 1] to [0, 1]
        rescaled = (sims + 1.0) / 2.0
        return {self.item_ids[i]: float(rescaled[i]) for i in range(len(self.item_ids))}

    def get_similar_articles(self, article_id: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Return Top-K most semantically similar articles to a reference article."""
        if article_id not in self.item_embeddings or self._embedding_matrix is None:
            return []

        art_vec = self.item_embeddings[article_id].reshape(1, -1)
        sims = np.dot(art_vec, self._embedding_matrix.T).squeeze(0)
        
        ranked_indices = np.argsort(-sims)
        results = []
        for idx in ranked_indices:
            candidate_id = self.item_ids[idx]
            if candidate_id != article_id:
                results.append((candidate_id, float((sims[idx] + 1.0) / 2.0)))
                if len(results) >= top_k:
                    break

        return results
