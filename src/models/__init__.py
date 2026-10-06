"""Recommendation Models Package.

Includes:
- TwoTowerRecommender: PyTorch Neural Collaborative Filtering / Two-Tower model
- ContentBasedRecommender: Cosine similarity on dense semantic article vectors
- HybridRecommendationEngine: Blended ranker combining CF + Semantic affinity + Fallbacks
"""

from .collaborative import TwoTowerRecommender, CollaborativeDataset
from .content_based import ContentBasedRecommender
from .hybrid_engine import HybridRecommendationEngine

__all__ = [
    "TwoTowerRecommender",
    "CollaborativeDataset",
    "ContentBasedRecommender",
    "HybridRecommendationEngine"
]
