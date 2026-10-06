"""FastAPI Endpoint Validation & Integration Tests."""

import os
import sys
import tempfile
import pytest
from fastapi.testclient import TestClient

# Ensure src is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Configure test environment variables before importing app
temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
os.environ["DATABASE_URL"] = f"sqlite:///{temp_db.name}"
os.environ["DATABASE_PATH"] = temp_db.name

from src.api.app import app
from src.database import DatabaseManager


@pytest.fixture(scope="module")
def client():
    """Create test client with initialized database fixture."""
    with TestClient(app) as test_client:
        # Seed test catalog
        db_mgr = DatabaseManager(db_url=os.environ["DATABASE_URL"])
        test_articles = [
            {
                "article_id": "ART_001",
                "category": "sports",
                "subcategory": "football",
                "title": "Championship Finals Preview",
                "abstract": "Key match details and starting lineups.",
                "url": "https://news.com/art_001",
                "embedding": [0.05] * 384
            },
            {
                "article_id": "ART_002",
                "category": "sports",
                "subcategory": "basketball",
                "title": "Playoffs Semifinal Game 7",
                "abstract": "Decisive clutch plays in final minutes.",
                "url": "https://news.com/art_002",
                "embedding": [0.06] * 384
            },
            {
                "article_id": "ART_003",
                "category": "technology",
                "subcategory": "ai",
                "title": "Breakthrough in Neural Embeddings",
                "abstract": "New transformer model benchmarks.",
                "url": "https://news.com/art_003",
                "embedding": [0.8] * 384
            }
        ]
        db_mgr.upsert_articles(test_articles)
        yield test_client

    if os.path.exists(temp_db.name):
        try:
            os.remove(temp_db.name)
        except Exception:
            pass


def test_health_endpoint(client):
    """Test healthcheck endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["healthy", "degraded"]
    assert "timestamp" in data
    assert "database_connected" in data


def test_interaction_logging(client):
    """Test POST /interaction logs implicit signal and returns composite affinity."""
    payload = {
        "user_id": "USER_TEST_101",
        "article_id": "ART_001",
        "clicked": 1,
        "dwell_time": 85.0,
        "scroll_depth": 0.90
    }
    response = client.post("/interaction", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert data["status"] == "success"
    assert data["user_id"] == "USER_TEST_101"
    assert data["article_id"] == "ART_001"
    assert 0.0 <= data["computed_affinity_score"] <= 1.0
    # Deep read should have relatively high affinity
    assert data["computed_affinity_score"] > 0.60


def test_interaction_validation_error(client):
    """Test POST /interaction rejects invalid scroll depth out of bounds."""
    bad_payload = {
        "user_id": "USER_TEST_101",
        "article_id": "ART_001",
        "clicked": 1,
        "dwell_time": 10.0,
        "scroll_depth": 2.5  # Invalid: > 1.0
    }
    response = client.post("/interaction", json=bad_payload)
    assert response.status_code == 422


def test_recommendation_cold_start(client):
    """Test GET /recommend/{user_id} provides popularity fallback for unknown reader."""
    response = client.get("/recommend/USER_NEW_UNKNOWN?k=2")
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == "USER_NEW_UNKNOWN"
    assert data["count"] <= 2
    assert len(data["recommendations"]) <= 2
    assert data["recommendation_mode"] == "cold_start_popularity"
    for rec in data["recommendations"]:
        assert "article_id" in rec
        assert "title" in rec
        assert "score" in rec
        assert "explanation" in rec


def test_recommendation_filtering(client):
    """Test consumed articles are filtered out from recommendation list."""
    # First log interaction with ART_001
    client.post("/interaction", json={
        "user_id": "USER_READER_55",
        "article_id": "ART_001",
        "clicked": 1,
        "dwell_time": 45.0,
        "scroll_depth": 0.70
    })

    # Request recommendations with filter_consumed=true
    response = client.get("/recommend/USER_READER_55?k=5&filter_consumed=true")
    assert response.status_code == 200
    data = response.json()
    rec_ids = [r["article_id"] for r in data["recommendations"]]
    assert "ART_001" not in rec_ids
