"""Production FastAPI Inference Service.

Serves low-latency personalized Top-K recommendations and ingests
streaming multi-modal interaction telemetry (click, dwell time, scroll depth).
"""

import os
import sys
import time
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np

# Ensure parent directory is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.database import DatabaseManager
from src.telemetry_processor import TelemetryProcessor
from src.models.collaborative import TwoTowerRecommender
from src.models.content_based import ContentBasedRecommender
from src.models.hybrid_engine import HybridRecommendationEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Global singletons
db_manager: Optional[DatabaseManager] = None
telemetry_processor: Optional[TelemetryProcessor] = None
hybrid_engine: Optional[HybridRecommendationEngine] = None
cf_model: Optional[TwoTowerRecommender] = None
content_model: Optional[ContentBasedRecommender] = None


# Pydantic Schemas
class InteractionRequest(BaseModel):
    user_id: str = Field(..., description="Unique identifier of the reader/user")
    article_id: str = Field(..., description="Identifier of the target article")
    clicked: int = Field(0, ge=0, le=1, description="Binary click indication: 0 or 1")
    dwell_time: float = Field(0.0, ge=0.0, description="Dwell/read duration in seconds")
    scroll_depth: float = Field(0.0, ge=0.0, le=1.0, description="Viewport scroll traversal percentage [0.0, 1.0]")
    timestamp: Optional[str] = Field(None, description="ISO formatted timestamp of event")

    model_config = {
        "json_schema_extra": {
            "example": {
                "user_id": "U00042",
                "article_id": "N00015",
                "clicked": 1,
                "dwell_time": 85.5,
                "scroll_depth": 0.92
            }
        }
    }


class InteractionResponse(BaseModel):
    status: str
    interaction_id: int
    user_id: str
    article_id: str
    computed_affinity_score: float
    message: str


class RecommendedArticle(BaseModel):
    article_id: str
    title: str
    category: str
    subcategory: Optional[str] = None
    url: Optional[str] = None
    score: float
    cf_score: Optional[float] = None
    content_score: Optional[float] = None
    explanation: str


class RecommendationResponse(BaseModel):
    user_id: str
    recommendation_mode: str
    count: int
    latency_ms: float
    recommendations: List[RecommendedArticle]


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    database_connected: bool
    cf_model_loaded: bool
    content_model_loaded: bool
    articles_count: int


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager: load feature store, neural models, and embeddings on startup."""
    global db_manager, telemetry_processor, hybrid_engine, cf_model, content_model

    logger.info("Initializing Content Recommendation Engine Service...")
    db_path = os.getenv("DATABASE_PATH", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "content_rec.db")))
    db_url = os.getenv("DATABASE_URL", f"sqlite:///{db_path}")

    db_manager = DatabaseManager(db_url=db_url)
    telemetry_processor = TelemetryProcessor(weight_click=0.35, weight_dwell=0.40, weight_scroll=0.25)

    # Load Two-Tower CF Model Checkpoint
    checkpoint_dir = os.getenv("CHECKPOINT_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "checkpoints")))
    cf_ckpt_path = os.path.join(checkpoint_dir, "two_tower_best.pt")
    if os.path.exists(cf_ckpt_path):
        try:
            cf_model = TwoTowerRecommender.load_checkpoint(cf_ckpt_path)
            logger.info("Two-Tower CF neural model checkpoint loaded.")
        except Exception as e:
            logger.warning(f"Failed to load Two-Tower checkpoint: {e}")
            cf_model = None
    else:
        logger.info(f"Two-Tower checkpoint not found at {cf_ckpt_path}. Operating in content/cold-start fallback mode.")
        cf_model = None

    # Load Semantic Content Embeddings
    emb_cache_path = os.path.join(checkpoint_dir, "content_embeddings.npz")
    if os.path.exists(emb_cache_path):
        try:
            loaded = np.load(emb_cache_path)
            emb_dict = {k: loaded[k] for k in loaded.files}
            content_model = ContentBasedRecommender(emb_dict)
            logger.info(f"Loaded {len(emb_dict)} article semantic embeddings into memory.")
        except Exception as e:
            logger.warning(f"Failed to load embeddings cache: {e}")
            content_model = None
    else:
        logger.info(f"Embeddings cache not found at {emb_cache_path}.")
        content_model = None

    # Assemble Hybrid Recommendation Engine
    hybrid_engine = HybridRecommendationEngine(
        collaborative_model=cf_model,
        content_model=content_model,
        db_manager=db_manager,
        alpha_cf=0.60,
        beta_category=0.10
    )

    logger.info("Engine components initialized and ready for inference.")
    yield
    logger.info("Shutting down Content Recommendation Engine Service...")


app = FastAPI(
    title="Personalized Content Recommendation & Affinity Engine API",
    description="Low-latency inference service serving neural collaborative & semantic hybrid recommendations.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def healthcheck():
    """Service liveness and readiness probe."""
    articles_count = 0
    db_connected = False
    if db_manager is not None:
        try:
            with db_manager.get_session() as session:
                from src.database import ArticleModel
                articles_count = session.query(ArticleModel).count()
                db_connected = True
        except Exception:
            db_connected = False

    return HealthResponse(
        status="healthy" if db_connected else "degraded",
        timestamp=datetime.utcnow().isoformat(),
        database_connected=db_connected,
        cf_model_loaded=cf_model is not None,
        content_model_loaded=content_model is not None,
        articles_count=articles_count
    )


@app.post("/interaction", response_model=InteractionResponse, status_code=status.HTTP_201_CREATED, tags=["Telemetry"])
def log_interaction(event: InteractionRequest):
    """Ingest real-time user implicit interaction signals and calculate composite affinity score."""
    if db_manager is None or telemetry_processor is None:
        raise HTTPException(status_code=500, detail="Service components not initialized.")

    # 1. Compute normalized composite affinity score
    affinity = telemetry_processor.calculate_single_affinity(
        clicked=event.clicked,
        dwell_time=event.dwell_time,
        scroll_depth=event.scroll_depth
    )

    event_time = None
    if event.timestamp:
        try:
            event_time = datetime.fromisoformat(event.timestamp)
        except Exception:
            event_time = datetime.utcnow()

    # 2. Persist to relational telemetry feature store
    try:
        interaction_record = db_manager.log_interaction(
            user_id=event.user_id,
            article_id=event.article_id,
            clicked=event.clicked,
            dwell_time=event.dwell_time,
            scroll_depth=event.scroll_depth,
            affinity_score=affinity,
            timestamp=event_time
        )
    except Exception as e:
        logger.error(f"Failed to log telemetry to DB: {e}")
        raise HTTPException(status_code=500, detail="Failed to log telemetry interaction.")

    # 3. Incrementally update user affinity profile in background / real-time
    try:
        recent_interactions = db_manager.get_user_interactions(event.user_id, limit=30)
        if recent_interactions:
            import pandas as pd
            df_hist = pd.DataFrame(recent_interactions)
            # Attach category if missing
            art_meta = db_manager.get_article(event.article_id)
            if art_meta and "category" not in df_hist.columns:
                df_hist["category"] = art_meta.get("category", "general")
            prof_dict = telemetry_processor.build_user_affinity_profiles(df_hist)
            if event.user_id in prof_dict:
                p = prof_dict[event.user_id]
                db_manager.upsert_user_affinity_profile(
                    user_id=event.user_id,
                    total_interactions=p["total_interactions"],
                    total_clicks=p["total_clicks"],
                    avg_dwell_time=p["avg_dwell_time"],
                    avg_scroll_depth=p["avg_scroll_depth"],
                    category_affinities=p["category_affinities"]
                )
    except Exception as e:
        logger.warning(f"Could not update user profile cache: {e}")

    return InteractionResponse(
        status="success",
        interaction_id=interaction_record["interaction_id"],
        user_id=event.user_id,
        article_id=event.article_id,
        computed_affinity_score=round(affinity, 4),
        message="Telemetry event ingested and affinity formulated successfully."
    )


@app.get("/recommend/{user_id}", response_model=RecommendationResponse, tags=["Inference"])
def get_recommendations(
    user_id: str,
    k: int = Query(10, ge=1, le=50, description="Number of recommendations to retrieve"),
    filter_consumed: bool = Query(True, description="Filter out articles the user has already consumed")
):
    """Retrieve personalized Top-K ranked articles with metadata and transparent affinity explanations."""
    if hybrid_engine is None:
        raise HTTPException(status_code=500, detail="Hybrid recommendation engine is not initialized.")

    start_time = time.perf_counter()

    # Determine recommendation mode
    user_history = db_manager.get_user_interactions(user_id, limit=1) if db_manager else []
    is_known_cf = (cf_model is not None and user_id in cf_model.user2idx)

    if is_known_cf:
        mode = "hybrid_two_tower_semantic"
    elif len(user_history) > 0:
        mode = "content_warm_start"
    else:
        mode = "cold_start_popularity"

    try:
        ranked_items = hybrid_engine.recommend(user_id=user_id, k=k, filter_consumed=filter_consumed)
    except Exception as e:
        logger.error(f"Inference error for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    latency_ms = (time.perf_counter() - start_time) * 1000.0

    # Format output items
    recs = [
        RecommendedArticle(
            article_id=item["article_id"],
            title=item["title"],
            category=item["category"],
            subcategory=item.get("subcategory"),
            url=item.get("url"),
            score=item["score"],
            cf_score=item.get("cf_score"),
            content_score=item.get("content_score"),
            explanation=item["explanation"]
        )
        for item in ranked_items
    ]

    return RecommendationResponse(
        user_id=user_id,
        recommendation_mode=mode,
        count=len(recs),
        latency_ms=round(latency_ms, 2),
        recommendations=recs
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
