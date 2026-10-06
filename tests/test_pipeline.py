"""Unit Tests for Telemetry Processor, Embeddings, Models, and Database Routines."""

import os
import sys
import tempfile
import pytest
import numpy as np
import pandas as pd
import torch

# Ensure src is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.telemetry_processor import TelemetryProcessor
from src.database import DatabaseManager, Base
from src.embeddings import ContentEmbeddingExtractor
from src.models.collaborative import TwoTowerRecommender, CollaborativeDataset
from src.models.content_based import ContentBasedRecommender
from src.models.hybrid_engine import HybridRecommendationEngine
from data.download_data import generate_synthetic_news, generate_synthetic_behaviors, simulate_interaction_telemetry


class TestTelemetryProcessor:
    """Validate mathematical formulation of composite affinity target scores."""

    def setup_method(self):
        self.processor = TelemetryProcessor(weight_click=0.35, weight_dwell=0.40, weight_scroll=0.25)

    def test_weight_normalization(self):
        assert abs((self.processor.w1 + self.processor.w2 + self.processor.w3) - 1.0) < 1e-6

    def test_affinity_score_bounds(self):
        # Minimum possible
        min_aff = self.processor.calculate_single_affinity(clicked=0, dwell_time=0.0, scroll_depth=0.0)
        assert min_aff == 0.0

        # Maximum possible
        max_aff = self.processor.calculate_single_affinity(clicked=1, dwell_time=300.0, scroll_depth=1.0)
        assert max_aff == 1.0

        # intermediate
        mid_aff = self.processor.calculate_single_affinity(clicked=1, dwell_time=45.0, scroll_depth=0.6)
        assert 0.0 <= mid_aff <= 1.0

    def test_bounce_vs_deep_read(self):
        # Bounce: clicked but quick leave
        bounce_aff = self.processor.calculate_single_affinity(clicked=1, dwell_time=3.0, scroll_depth=0.15)
        # Deep read: clicked and long read with full scroll
        deep_read_aff = self.processor.calculate_single_affinity(clicked=1, dwell_time=120.0, scroll_depth=0.95)
        # Non-click impression: glanced past
        glance_aff = self.processor.calculate_single_affinity(clicked=0, dwell_time=2.0, scroll_depth=0.05)

        assert glance_aff < bounce_aff
        assert bounce_aff < deep_read_aff

    def test_dataframe_processing(self):
        raw_df = pd.DataFrame([
            {"user_id": "U1", "article_id": "N1", "clicked": 1, "dwell_time": 60.0, "scroll_depth": 0.8},
            {"user_id": "U1", "article_id": "N2", "clicked": 0, "dwell_time": 1.5, "scroll_depth": 0.05},
        ])
        processed = self.processor.process_telemetry_dataframe(raw_df)
        assert "affinity_score" in processed.columns
        assert "normalized_dwell" in processed.columns
        assert processed.loc[0, "affinity_score"] > processed.loc[1, "affinity_score"]


class TestDatabaseManager:
    """Validate SQLite relational feature store operations."""

    def setup_method(self):
        self.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.db_path = self.temp_db.name
        self.db_url = f"sqlite:///{self.db_path}"
        self.db_manager = DatabaseManager(db_url=self.db_url)

    def teardown_method(self):
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_articles_upsert_and_fetch(self):
        sample_articles = [
            {
                "article_id": "N100",
                "category": "sports",
                "subcategory": "football",
                "title": "Championship Match Preview",
                "abstract": "In-depth tactical analysis.",
                "url": "http://news/n100",
                "embedding": [0.1] * 384
            },
            {
                "article_id": "N200",
                "category": "finance",
                "subcategory": "markets",
                "title": "Stock Market Update",
                "abstract": "Indices rise amid strong earnings.",
                "url": "http://news/n200",
                "embedding": [0.2] * 384
            }
        ]
        count = self.db_manager.upsert_articles(sample_articles)
        assert count == 2

        art = self.db_manager.get_article("N100")
        assert art is not None
        assert art["title"] == "Championship Match Preview"
        assert len(art["embedding"]) == 384

    def test_interaction_logging_and_popularity(self):
        # Log interactions
        self.db_manager.log_interaction("U1", "N100", clicked=1, dwell_time=90.0, scroll_depth=0.9, affinity_score=0.88)
        self.db_manager.log_interaction("U2", "N100", clicked=1, dwell_time=70.0, scroll_depth=0.8, affinity_score=0.80)
        self.db_manager.log_interaction("U1", "N200", clicked=0, dwell_time=2.0, scroll_depth=0.1, affinity_score=0.03)

        history = self.db_manager.get_user_interactions("U1")
        assert len(history) == 2

        popular = self.db_manager.get_popular_articles(limit=5)
        assert len(popular) >= 1
        assert popular[0]["article_id"] == "N100"
        assert popular[0]["popularity_clicks"] == 2


class TestEmbeddingsAndModels:
    """Validate semantic vector extraction and Neural Collaborative Filtering."""

    def test_content_embedding_extractor(self):
        extractor = ContentEmbeddingExtractor()
        texts = ["Artificial Intelligence neural network model", "Quarterback passes football for touchdown"]
        embeddings = extractor.encode_texts(texts)

        assert embeddings.shape == (2, 384)
        # Vectors should be L2-normalized
        norm_0 = np.linalg.norm(embeddings[0])
        assert abs(norm_0 - 1.0) < 1e-4

        sim_matrix = ContentEmbeddingExtractor.cosine_similarity_matrix(embeddings[0], embeddings[1])
        assert -1.0 <= sim_matrix[0, 0] <= 1.0

    def test_two_tower_recommender(self):
        num_users = 10
        num_items = 20
        model = TwoTowerRecommender(num_users=num_users, num_items=num_items, embedding_dim=32)

        user_tensor = torch.tensor([0, 1], dtype=torch.long)
        item_tensor = torch.tensor([5, 8], dtype=torch.long)
        preds = model(user_tensor, item_tensor)

        assert preds.shape == (2,)
        assert (preds >= 0.0).all() and (preds <= 1.0).all()

        u_vec = model.get_user_vector(0)
        assert u_vec.shape == (32,)

        all_item_vecs = model.get_all_item_embeddings()
        assert all_item_vecs.shape == (num_items, 32)

    def test_hybrid_engine_recommendations(self):
        temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        db_path = temp_db.name
        db_mgr = DatabaseManager(db_url=f"sqlite:///{db_path}")

        articles = [
            {"article_id": "N1", "category": "sports", "title": "Game 1", "embedding": [0.1] * 384},
            {"article_id": "N2", "category": "sports", "title": "Game 2", "embedding": [0.1] * 384},
            {"article_id": "N3", "category": "finance", "title": "Market rally", "embedding": [0.5] * 384},
        ]
        db_mgr.upsert_articles(articles)
        db_mgr.log_interaction("U_test", "N1", clicked=1, dwell_time=100.0, scroll_depth=0.9, affinity_score=0.9)

        content_model = ContentBasedRecommender({
            "N1": np.ones(384, dtype=np.float32) / np.sqrt(384),
            "N2": np.ones(384, dtype=np.float32) / np.sqrt(384),
            "N3": -np.ones(384, dtype=np.float32) / np.sqrt(384)
        })

        hybrid = HybridRecommendationEngine(
            collaborative_model=None,
            content_model=content_model,
            db_manager=db_mgr
        )

        recs = hybrid.recommend("U_test", k=2, filter_consumed=True)
        assert len(recs) == 2
        # N1 should be filtered out because it was already consumed
        assert all(r["article_id"] != "N1" for r in recs)
        # N2 (sports) should be ranked higher than N3 (finance) due to semantic similarity to N1
        assert recs[0]["article_id"] == "N2"

        os.remove(db_path)
