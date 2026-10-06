"""Relational Feature Store and SQLite/PostgreSQL Database Layer.

Provides SQLAlchemy declarative ORM models and high-throughput query interfaces
for raw telemetry logs, processed user affinity vectors, and article metadata.
"""

import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Optional, Any, Union
from contextlib import contextmanager

from sqlalchemy import (
    create_engine, Column, String, Integer, Float, Text, DateTime, ForeignKey, Index
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship, Session

logger = logging.getLogger(__name__)

Base = declarative_base()


class ArticleModel(Base):
    """Article catalog metadata and precomputed semantic topic embeddings."""
    __tablename__ = "articles"

    article_id = Column(String(64), primary_key=True)
    category = Column(String(64), nullable=False, index=True)
    subcategory = Column(String(64), nullable=True)
    title = Column(Text, nullable=False)
    abstract = Column(Text, nullable=True)
    url = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    embedding_json = Column(Text, nullable=True)

    interactions = relationship("InteractionModel", back_populates="article", cascade="all, delete-orphan")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "article_id": self.article_id,
            "category": self.category,
            "subcategory": self.subcategory,
            "title": self.title,
            "abstract": self.abstract,
            "url": self.url,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "embedding": json.loads(self.embedding_json) if self.embedding_json else None
        }


class UserModel(Base):
    """Registered or guest reader identities."""
    __tablename__ = "users"

    user_id = Column(String(64), primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_active = Column(DateTime, default=datetime.utcnow)

    interactions = relationship("InteractionModel", back_populates="user", cascade="all, delete-orphan")
    profile = relationship("UserAffinityProfileModel", back_populates="user", uselist=False, cascade="all, delete-orphan")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_active": self.last_active.isoformat() if self.last_active else None
        }


class InteractionModel(Base):
    """Implicit interaction telemetry store."""
    __tablename__ = "interactions"

    interaction_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(64), ForeignKey("users.user_id"), nullable=False, index=True)
    article_id = Column(String(64), ForeignKey("articles.article_id"), nullable=False, index=True)
    clicked = Column(Integer, default=0, nullable=False)
    dwell_time = Column(Float, default=0.0, nullable=False)
    scroll_depth = Column(Float, default=0.0, nullable=False)
    affinity_score = Column(Float, default=0.0, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

    user = relationship("UserModel", back_populates="interactions")
    article = relationship("ArticleModel", back_populates="interactions")

    __table_args__ = (
        Index("idx_interactions_composite", "user_id", "article_id"),
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "interaction_id": self.interaction_id,
            "user_id": self.user_id,
            "article_id": self.article_id,
            "clicked": self.clicked,
            "dwell_time": self.dwell_time,
            "scroll_depth": self.scroll_depth,
            "affinity_score": self.affinity_score,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None
        }


class UserAffinityProfileModel(Base):
    """Aggregated reader category affinities and latent user representation."""
    __tablename__ = "user_affinity_profiles"

    user_id = Column(String(64), ForeignKey("users.user_id"), primary_key=True)
    total_interactions = Column(Integer, default=0)
    total_clicks = Column(Integer, default=0)
    avg_dwell_time = Column(Float, default=0.0)
    avg_scroll_depth = Column(Float, default=0.0)
    category_affinities_json = Column(Text, nullable=True)
    latent_vector_json = Column(Text, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("UserModel", back_populates="profile")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "total_interactions": self.total_interactions,
            "total_clicks": self.total_clicks,
            "avg_dwell_time": self.avg_dwell_time,
            "avg_scroll_depth": self.avg_scroll_depth,
            "category_affinities": json.loads(self.category_affinities_json) if self.category_affinities_json else {},
            "latent_vector": json.loads(self.latent_vector_json) if self.latent_vector_json else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }


class DatabaseManager:
    """Manages database connection lifecycle, pooling, and telemetry feature queries."""

    def __init__(self, db_url: Optional[str] = None):
        if not db_url:
            default_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "content_rec.db"))
            db_url = os.getenv("DATABASE_URL", f"sqlite:///{default_path}")
            
        self.db_url = db_url
        connect_args = {"check_same_thread": False} if self.db_url.startswith("sqlite") else {}
        self.engine = create_engine(self.db_url, connect_args=connect_args, pool_pre_ping=True)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self.init_schema()

    def init_schema(self) -> None:
        """Create tables if they don't already exist."""
        Base.metadata.create_all(bind=self.engine)
        logger.info(f"Database schema initialized against {self.db_url}")

    @contextmanager
    def get_session(self):
        """Provide a transactional scope around a series of operations."""
        session: Session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def upsert_user(self, user_id: str) -> None:
        """Ensure user record exists in users table."""
        with self.get_session() as session:
            user = session.query(UserModel).filter_by(user_id=user_id).first()
            if not user:
                user = UserModel(user_id=user_id, last_active=datetime.utcnow())
                session.add(user)
            else:
                user.last_active = datetime.utcnow()

    def upsert_articles(self, articles_data: List[Dict[str, Any]]) -> int:
        """Batch insert or update article records."""
        count = 0
        with self.get_session() as session:
            for item in articles_data:
                art_id = item["article_id"]
                art = session.query(ArticleModel).filter_by(article_id=art_id).first()
                emb_json = json.dumps(item["embedding"]) if "embedding" in item and item["embedding"] is not None else None
                
                if not art:
                    art = ArticleModel(
                        article_id=art_id,
                        category=item["category"],
                        subcategory=item.get("subcategory"),
                        title=item["title"],
                        abstract=item.get("abstract"),
                        url=item.get("url"),
                        embedding_json=emb_json
                    )
                    session.add(art)
                else:
                    art.category = item["category"]
                    art.subcategory = item.get("subcategory")
                    art.title = item["title"]
                    art.abstract = item.get("abstract")
                    art.url = item.get("url")
                    if emb_json:
                        art.embedding_json = emb_json
                count += 1
        return count

    def log_interaction(
        self,
        user_id: str,
        article_id: str,
        clicked: int,
        dwell_time: float,
        scroll_depth: float,
        affinity_score: float,
        timestamp: Optional[datetime] = None
    ) -> InteractionModel:
        """Record real-time telemetry interaction event and update user last active."""
        ts = timestamp or datetime.utcnow()
        with self.get_session() as session:
            # Ensure user exists
            user = session.query(UserModel).filter_by(user_id=user_id).first()
            if not user:
                user = UserModel(user_id=user_id, last_active=ts)
                session.add(user)
            else:
                user.last_active = ts

            # Ensure article exists (create minimal stub if unknown)
            article = session.query(ArticleModel).filter_by(article_id=article_id).first()
            if not article:
                article = ArticleModel(
                    article_id=article_id,
                    category="general",
                    title="Unknown Article",
                    abstract="Auto-registered during telemetry ingestion"
                )
                session.add(article)

            interaction = InteractionModel(
                user_id=user_id,
                article_id=article_id,
                clicked=clicked,
                dwell_time=dwell_time,
                scroll_depth=scroll_depth,
                affinity_score=affinity_score,
                timestamp=ts
            )
            session.add(interaction)
            session.flush()
            interaction_dict = interaction.to_dict()
            return interaction_dict

    def get_article(self, article_id: str) -> Optional[Dict[str, Any]]:
        """Fetch article by ID."""
        with self.get_session() as session:
            art = session.query(ArticleModel).filter_by(article_id=article_id).first()
            return art.to_dict() if art else None

    def get_all_articles(self) -> List[Dict[str, Any]]:
        """Fetch all articles from feature store."""
        with self.get_session() as session:
            articles = session.query(ArticleModel).all()
            return [a.to_dict() for a in articles]

    def get_popular_articles(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Popularity-based fallback for cold-start recommendations."""
        from sqlalchemy import func
        with self.get_session() as session:
            # Rank by click volume + average composite affinity
            subquery = (
                session.query(
                    InteractionModel.article_id,
                    func.count(InteractionModel.interaction_id).label("total_interactions"),
                    func.sum(InteractionModel.clicked).label("clicks"),
                    func.avg(InteractionModel.affinity_score).label("avg_affinity")
                )
                .group_by(InteractionModel.article_id)
                .subquery()
            )
            results = (
                session.query(ArticleModel, subquery.c.clicks, subquery.c.avg_affinity)
                .outerjoin(subquery, ArticleModel.article_id == subquery.c.article_id)
                .order_by(subquery.c.clicks.desc().nullslast(), ArticleModel.created_at.desc())
                .limit(limit)
                .all()
            )
            out = []
            for art, clicks, avg_aff in results:
                d = art.to_dict()
                d["popularity_clicks"] = clicks or 0
                d["avg_affinity"] = float(avg_aff or 0.0)
                out.append(d)
            return out

    def get_user_interactions(self, user_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Fetch historical interactions for a given user."""
        with self.get_session() as session:
            rows = (
                session.query(InteractionModel)
                .filter_by(user_id=user_id)
                .order_by(InteractionModel.timestamp.desc())
                .limit(limit)
                .all()
            )
            return [r.to_dict() for r in rows]

    def upsert_user_affinity_profile(
        self,
        user_id: str,
        total_interactions: int,
        total_clicks: int,
        avg_dwell_time: float,
        avg_scroll_depth: float,
        category_affinities: Dict[str, float],
        latent_vector: Optional[List[float]] = None
    ) -> None:
        """Upsert aggregated user affinity profile."""
        with self.get_session() as session:
            prof = session.query(UserAffinityProfileModel).filter_by(user_id=user_id).first()
            cat_json = json.dumps(category_affinities)
            latent_json = json.dumps(latent_vector) if latent_vector is not None else None
            
            if not prof:
                prof = UserAffinityProfileModel(
                    user_id=user_id,
                    total_interactions=total_interactions,
                    total_clicks=total_clicks,
                    avg_dwell_time=avg_dwell_time,
                    avg_scroll_depth=avg_scroll_depth,
                    category_affinities_json=cat_json,
                    latent_vector_json=latent_json,
                    updated_at=datetime.utcnow()
                )
                session.add(prof)
            else:
                prof.total_interactions = total_interactions
                prof.total_clicks = total_clicks
                prof.avg_dwell_time = avg_dwell_time
                prof.avg_scroll_depth = avg_scroll_depth
                prof.category_affinities_json = cat_json
                if latent_json:
                    prof.latent_vector_json = latent_json
                prof.updated_at = datetime.utcnow()

    def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Fetch user affinity profile."""
        with self.get_session() as session:
            prof = session.query(UserAffinityProfileModel).filter_by(user_id=user_id).first()
            return prof.to_dict() if prof else None
