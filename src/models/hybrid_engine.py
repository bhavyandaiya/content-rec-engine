"""Hybrid Recommendation Engine.

Blends Neural Collaborative Filtering (Two-Tower PyTorch) with
Content-Based Semantic Embeddings and implicit category affinity priors.
Includes dynamic cold-start routing and candidate re-ranking.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple, Set

import numpy as np
import pandas as pd

from .collaborative import TwoTowerRecommender
from .content_based import ContentBasedRecommender
from ..database import DatabaseManager

logger = logging.getLogger(__name__)


class HybridRecommendationEngine:
    """Blended recommendation ranker combining latent collaborative embeddings,
    dense semantic content vectors, and implicit interaction telemetry.
    """

    def __init__(
        self,
        collaborative_model: Optional[TwoTowerRecommender] = None,
        content_model: Optional[ContentBasedRecommender] = None,
        db_manager: Optional[DatabaseManager] = None,
        alpha_cf: float = 0.60,
        beta_category: float = 0.10
    ):
        """Initialize hybrid engine with configurable blending parameters.

        Args:
            collaborative_model: Pre-trained TwoTowerRecommender instance.
            content_model: Pre-computed ContentBasedRecommender instance.
            db_manager: Relational database manager for telemetry and catalog.
            alpha_cf: Weight allocated to neural CF score (vs 1 - alpha for content score).
            beta_category: Extra affinity boost for articles in preferred user categories.
        """
        self.cf_model = collaborative_model
        self.content_model = content_model
        self.db = db_manager
        self.alpha_cf = alpha_cf
        self.beta_category = beta_category

    def recommend(
        self,
        user_id: str,
        k: int = 10,
        filter_consumed: bool = True,
        user_history: Optional[List[Dict[str, Any]]] = None,
        excluded_ids: Optional[Set[str]] = None
    ) -> List[Dict[str, Any]]:
        """Serve Top-K personalized recommendations for a given reader.
        
        Handles:
        1. Full Hybrid (known user with CF embedding + semantic history)
        2. Content-Only Warm Start (new user with few real-time session reads)
        3. Popularity Fallback (completely cold-start user)
        """
        if self.db is None:
            raise RuntimeError("Database manager is not initialized in HybridEngine.")

        # 1. Fetch user history & profile
        if user_history is None:
            user_history = self.db.get_user_interactions(user_id, limit=50)
            
        user_profile = self.db.get_user_profile(user_id)
        
        if excluded_ids is not None:
            consumed_article_ids = set(excluded_ids)
        elif filter_consumed:
            consumed_article_ids = {event["article_id"] for event in user_history}
        else:
            consumed_article_ids = set()

        # 2. Check user status for Collaborative Model
        is_known_cf_user = (
            self.cf_model is not None and
            user_id in self.cf_model.user2idx
        )

        # 3. Handle Completely Cold-Start User (no history)
        if not is_known_cf_user and len(user_history) == 0:
            logger.info(f"User '{user_id}' has no interaction history. Falling back to popularity baseline.")
            return self._recommend_cold_start(k=k, exclude_ids=consumed_article_ids)

        # 4. Fetch candidate catalog
        all_articles = self.db.get_all_articles()
        if not all_articles:
            return []

        # 5. Compute CF Scores
        cf_scores: Dict[str, float] = {}
        if is_known_cf_user:
            u_idx = self.cf_model.user2idx[user_id]
            raw_cf_preds = self.cf_model.predict_all_for_user(u_idx)
            for art_idx, art_id in self.cf_model.idx2item.items():
                cf_scores[art_id] = float(raw_cf_preds[art_idx])

        # 6. Compute Content-Based Semantic Scores
        content_scores: Dict[str, float] = {}
        if self.content_model is not None and len(user_history) > 0:
            u_content_vec = self.content_model.build_user_vector_from_history(user_history)
            if u_content_vec is not None:
                content_scores = self.content_model.score_all_items_for_user(u_content_vec)

        # 7. Category Prior Scores
        cat_affinities = user_profile.get("category_affinities", {}) if user_profile else {}

        # 8. Blend and Rank Candidates
        candidate_ranks = []
        # Dynamic weighting based on interaction depth
        effective_alpha = self.alpha_cf if is_known_cf_user else 0.0

        for art in all_articles:
            art_id = art["article_id"]
            if art_id in consumed_article_ids:
                continue

            cf_score = cf_scores.get(art_id, 0.5)
            content_score = content_scores.get(art_id, 0.5)
            cat_boost = cat_affinities.get(art.get("category", ""), 0.0)

            # Blended scoring formula
            if is_known_cf_user and content_scores:
                hybrid_score = (effective_alpha * cf_score) + ((1.0 - effective_alpha) * content_score)
            elif is_known_cf_user:
                hybrid_score = cf_score
            elif content_scores:
                hybrid_score = content_score
            else:
                hybrid_score = 0.5

            # Apply category preference boost
            hybrid_score += (self.beta_category * cat_boost)
            hybrid_score = float(np.clip(hybrid_score, 0.0, 1.0))

            explanation = self._generate_explanation(art, cf_score, content_score, cat_boost, is_known_cf_user)

            candidate_ranks.append({
                "article_id": art_id,
                "title": art["title"],
                "category": art["category"],
                "subcategory": art.get("subcategory"),
                "url": art.get("url"),
                "score": round(hybrid_score, 4),
                "cf_score": round(cf_score, 4) if is_known_cf_user else None,
                "content_score": round(content_score, 4) if content_scores else None,
                "explanation": explanation
            })

        # Sort descending by composite hybrid score
        candidate_ranks.sort(key=lambda x: x["score"], reverse=True)
        return candidate_ranks[:k]

    def _recommend_cold_start(self, k: int = 10, exclude_ids: Optional[set] = None) -> List[Dict[str, Any]]:
        """Serve diverse popular articles for brand-new users."""
        popular_items = self.db.get_popular_articles(limit=k * 2)
        exclude_ids = exclude_ids or set()
        
        results = []
        seen_categories = set()
        
        # Round 1: ensure category diversity
        for item in popular_items:
            art_id = item["article_id"]
            if art_id in exclude_ids:
                continue
            cat = item.get("category")
            if cat not in seen_categories:
                seen_categories.add(cat)
                results.append({
                    "article_id": art_id,
                    "title": item["title"],
                    "category": item["category"],
                    "subcategory": item.get("subcategory"),
                    "url": item.get("url"),
                    "score": round(float(item.get("avg_affinity", 0.5)), 4),
                    "cf_score": None,
                    "content_score": None,
                    "explanation": f"Trending in {cat.capitalize()}"
                })
            if len(results) >= k:
                return results

        # Round 2: fill remaining slots
        for item in popular_items:
            art_id = item["article_id"]
            if art_id in exclude_ids or any(r["article_id"] == art_id for r in results):
                continue
            results.append({
                "article_id": art_id,
                "title": item["title"],
                "category": item["category"],
                "subcategory": item.get("subcategory"),
                "url": item.get("url"),
                "score": round(float(item.get("avg_affinity", 0.5)), 4),
                "cf_score": None,
                "content_score": None,
                "explanation": "Trending across platform"
            })
            if len(results) >= k:
                break

        return results

    def _generate_explanation(
        self,
        article: Dict[str, Any],
        cf_score: float,
        content_score: float,
        cat_boost: float,
        is_known_cf: bool
    ) -> str:
        """Produce transparent rationale for recommendation card."""
        cat = article.get("category", "topics").capitalize()
        if cat_boost > 0.15:
            return f"Aligned with your high affinity for {cat} stories."
        elif content_score > 0.70:
            return f"Semantically similar to your recent reads in {cat}."
        elif is_known_cf and cf_score > 0.70:
            return f"Popular among readers with reading patterns similar to yours."
        else:
            return f"Recommended for you in {cat}."
