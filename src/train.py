"""Model Training Pipeline with Checkpointing & Feature Store Ingestion.

Trains the PyTorch Two-Tower collaborative network on implicit telemetry targets,
computes dense semantic transformer embeddings, and populates the SQLite feature store.
"""

import os
import sys
import json
import logging
import argparse
from typing import Dict, Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import pandas as pd

# Add repo root to python path if executed standalone
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.database import DatabaseManager
from src.telemetry_processor import TelemetryProcessor
from src.embeddings import ContentEmbeddingExtractor
from src.models.collaborative import TwoTowerRecommender, CollaborativeDataset
from data.download_data import ensure_dataset, simulate_interaction_telemetry

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_training_data(
    telemetry_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int], Dict[str, int]]:
    """Convert string IDs to continuous integer indices for PyTorch embeddings."""
    unique_users = sorted(telemetry_df["user_id"].unique())
    unique_items = sorted(telemetry_df["article_id"].unique())

    user2idx = {uid: i for i, uid in enumerate(unique_users)}
    item2idx = {iid: i for i, iid in enumerate(unique_items)}

    user_indices = np.array([user2idx[uid] for uid in telemetry_df["user_id"].values], dtype=np.int64)
    item_indices = np.array([item2idx[iid] for iid in telemetry_df["article_id"].values], dtype=np.int64)
    affinity_targets = np.array(telemetry_df["affinity_score"].values, dtype=np.float32)

    return user_indices, item_indices, affinity_targets, user2idx, item2idx


def train_collaborative_model(
    user_indices: np.ndarray,
    item_indices: np.ndarray,
    targets: np.ndarray,
    user2idx: Dict[str, int],
    item2idx: Dict[str, int],
    epochs: int = 15,
    batch_size: int = 256,
    lr: float = 0.001,
    embedding_dim: int = 64,
    checkpoint_path: str = "./checkpoints/two_tower_best.pt"
) -> TwoTowerRecommender:
    """Train Two-Tower neural model with early stopping on validation loss."""
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    logger.info(f"Training Two-Tower Recommender on device: {device}")

    dataset = CollaborativeDataset(user_indices, item_indices, targets)
    val_size = int(len(dataset) * 0.15)
    train_size = len(dataset) - val_size

    train_data, val_data = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    model = TwoTowerRecommender(
        num_users=len(user2idx),
        num_items=len(item2idx),
        embedding_dim=embedding_dim
    ).to(device)
    model.set_id_mappings(user2idx, item2idx)

    # Use Mean Squared Error to fit continuous affinity target score [0, 1]
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_val_loss = float("inf")
    os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for u_batch, i_batch, t_batch in train_loader:
            u_batch = u_batch.to(device)
            i_batch = i_batch.to(device)
            t_batch = t_batch.to(device)

            optimizer.zero_grad()
            preds = model(u_batch, i_batch)
            loss = criterion(preds, t_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(u_batch)

        train_loss /= train_size

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for u_batch, i_batch, t_batch in val_loader:
                u_batch = u_batch.to(device)
                i_batch = i_batch.to(device)
                t_batch = t_batch.to(device)
                preds = model(u_batch, i_batch)
                loss = criterion(preds, t_batch)
                val_loss += loss.item() * len(u_batch)

        val_loss /= val_size
        scheduler.step(val_loss)

        logger.info(f"Epoch {epoch:02d}/{epochs:02d} | Train MSE: {train_loss:.5f} | Val MSE: {val_loss:.5f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_checkpoint(checkpoint_path)

    # Reload best model
    best_model = TwoTowerRecommender.load_checkpoint(checkpoint_path, map_location=str(device))
    return best_model


def run_pipeline(
    data_dir: str = "./data",
    checkpoint_dir: str = "./checkpoints",
    db_path: str = "./data/content_rec.db",
    force_synthetic: bool = False,
    epochs: int = 12
) -> None:
    """Execute end-to-end data ingestion, embedding computation, and model training."""
    logger.info("Step 1: Ingesting dataset...")
    df_news, df_behaviors = ensure_dataset(data_dir=data_dir, fallback_only=force_synthetic)

    logger.info("Step 2: Simulating and processing implicit interaction telemetry...")
    raw_telemetry = simulate_interaction_telemetry(df_behaviors, df_news)
    telemetry_proc = TelemetryProcessor(weight_click=0.35, weight_dwell=0.40, weight_scroll=0.25)
    processed_telemetry = telemetry_proc.process_telemetry_dataframe(raw_telemetry)

    logger.info("Step 3: Extracting semantic text embeddings via all-MiniLM-L6-v2...")
    embedder = ContentEmbeddingExtractor()
    dense_embeddings = embedder.encode_articles(df_news)
    logger.info(f"Generated semantic vectors of shape {dense_embeddings.shape}")

    # Map embeddings to articles
    art_dict_list = []
    emb_dict = {}
    for idx, row in df_news.iterrows():
        aid = row["news_id"]
        vec = dense_embeddings[idx].tolist()
        emb_dict[aid] = dense_embeddings[idx]
        art_dict_list.append({
            "article_id": aid,
            "category": row["category"],
            "subcategory": row["subcategory"],
            "title": row["title"],
            "abstract": row["abstract"],
            "url": row["url"],
            "embedding": vec
        })

    # Save embeddings dictionary for fast inference
    os.makedirs(checkpoint_dir, exist_ok=True)
    emb_cache_path = os.path.join(checkpoint_dir, "content_embeddings.npz")
    np.savez_compressed(emb_cache_path, **{k: v for k, v in emb_dict.items()})
    logger.info(f"Persisted semantic embeddings cache to {emb_cache_path}")

    logger.info("Step 4: Initializing and populating Relational Feature Store...")
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    db_manager = DatabaseManager(db_url=f"sqlite:///{os.path.abspath(db_path)}")
    
    # Ingest articles
    db_manager.upsert_articles(art_dict_list)

    # Ingest unique users first
    unique_user_ids = sorted(processed_telemetry["user_id"].unique())
    logger.info(f"Ingesting {len(unique_user_ids)} unique users into feature store...")
    with db_manager.get_session() as session:
        from src.database import UserModel, InteractionModel
        existing_users = {u.user_id for u in session.query(UserModel).all()}
        new_users = [UserModel(user_id=uid) for uid in unique_user_ids if uid not in existing_users]
        if new_users:
            session.add_all(new_users)

    # Bulk insert telemetry interactions in efficient batches
    logger.info(f"Writing {len(processed_telemetry)} telemetry records to feature store...")
    batch_size = 2000
    interaction_records = []
    for _, row in processed_telemetry.iterrows():
        interaction_records.append(InteractionModel(
            user_id=row["user_id"],
            article_id=row["article_id"],
            clicked=int(row["clicked"]),
            dwell_time=float(row["dwell_time"]),
            scroll_depth=float(row["scroll_depth"]),
            affinity_score=float(row["affinity_score"])
        ))

    with db_manager.get_session() as session:
        for i in range(0, len(interaction_records), batch_size):
            session.bulk_save_objects(interaction_records[i:i + batch_size])

    # Compute and persist user affinity profiles
    logger.info("Computing user affinity profiles...")
    user_profiles = telemetry_proc.build_user_affinity_profiles(processed_telemetry)
    for uid, prof in user_profiles.items():
        db_manager.upsert_user_affinity_profile(
            user_id=uid,
            total_interactions=prof["total_interactions"],
            total_clicks=prof["total_clicks"],
            avg_dwell_time=prof["avg_dwell_time"],
            avg_scroll_depth=prof["avg_scroll_depth"],
            category_affinities=prof["category_affinities"]
        )

    logger.info("Step 5: Training PyTorch Two-Tower collaborative recommender...")
    u_idx, i_idx, targets, u2i, i2i = build_training_data(processed_telemetry)
    ckpt_path = os.path.join(checkpoint_dir, "two_tower_best.pt")
    train_collaborative_model(
        user_indices=u_idx,
        item_indices=i_idx,
        targets=targets,
        user2idx=u2i,
        item2idx=i2i,
        epochs=epochs,
        checkpoint_path=ckpt_path
    )

    logger.info("Pipeline execution completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run complete model training & feature store ingestion.")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    parser.add_argument("--db-path", type=str, default="./data/content_rec.db")
    parser.add_argument("--force-synthetic", action="store_true")
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()

    run_pipeline(
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        db_path=args.db_path,
        force_synthetic=args.force_synthetic,
        epochs=args.epochs
    )
