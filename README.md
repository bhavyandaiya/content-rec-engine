# Personalized Content Recommendation & Affinity Engine

An end-to-end recommendation and implicit user affinity engine built with PyTorch, Sentence-Transformers, and FastAPI. The system implements a PyTorch Two-Tower collaborative filtering architecture paired with Sentence-Transformers semantic text embeddings (`all-MiniLM-L6-v2`), driven by an implicit interaction telemetry pipeline (capturing click-through, log-saturated dwell time, and viewport scroll depth). Features and interaction events are persisted in a high-performance SQLite feature store, while low-latency personalized recommendations and real-time telemetry ingestion are served via FastAPI.

---

## Architecture

```text
+-----------------------------------------------------------------------------+
|                               Client / Reader                               |
|   - Reader Session Interactions (Clicks, Dwell Duration, Scroll Depth %)    |
+-----------------------------------------------------------------------------+
                                      |
                                      v [POST /interaction]
+-----------------------------------------------------------------------------+
|                             FastAPI Ingestion                               |
+-----------------------------------------------------------------------------+
                                      |
                                      v
+-----------------------------------------------------------------------------+
|                            Telemetry Processor                              |
|   - Normalized Dwell Time (Log Saturation)                                  |
|   - Continuous Composite Affinity Formulation                               |
|   - Temporal Recency Decay                                                  |
+-----------------------------------------------------------------------------+
                                      |
                                      v
+-----------------------------------------------------------------------------+
|                            SQLite Feature Store                             |
|   - Articles Catalog & Metadata                                             |
|   - Raw & Processed Interaction Telemetry                                   |
|   - Aggregated User Affinity Profiles                                       |
+-----------------------------------------------------------------------------+
               |                                             |
               v                                             v
+-----------------------------+               +-------------------------------+
|     PyTorch Two-Tower CF    |               |      Sentence Transformers    |
|  - User & Item Embeddings   |               |  - all-MiniLM-L6-v2 (384-dim) |
|  - Latent Representation    |               |  - User Semantic History      |
+-----------------------------+               +-------------------------------+
               \                                             /
                \                                           /
                 v                                         v
+-----------------------------------------------------------------------------+
|                           Hybrid Ranking Engine                             |
|   - Dynamic Fallback Routing (Cold Start / Warm Start / Full Hybrid)        |
|   - Blended Scoring: CF (60%) + Semantic (30%) + Category Affinity (10%)    |
|   - History Deduplication & Explanation Generation                          |
+-----------------------------------------------------------------------------+
                                      |
                                      v [GET /recommend/{user_id}]
+-----------------------------------------------------------------------------+
|                           FastAPI Inference Service                         |
|   - Low-Latency Ranked Top-K Response with Transparent Explanations         |
+-----------------------------------------------------------------------------+
```

---

## Composite Affinity Score Formulation

Traditional recommendation systems treat user engagement as a binary click (0 or 1). However, binary clicks frequently introduce clickbait noise and ignore actual reading engagement. This engine formulates a continuous affinity target score between `0.0` and `1.0` by fusing explicit click actions with continuous implicit telemetry signals:

`Affinity = (w1 * Click) + (w2 * D_norm) + (w3 * Scroll)`

### Component Definitions

- `Click`: Binary click indicator (`1` if the user opened the article, `0` otherwise).
- `Scroll`: Continuous vertical viewport scroll depth percentage (`0.0` to `1.0`).
- `D_norm`: Normalized reading dwell duration in seconds, scaled using a logarithmic saturation function:

`D_norm(t) = ln(1 + min(t, tau)) / ln(1 + tau)`

Where `tau = 180` seconds (3-minute reading saturation threshold representing diminishing marginal interest beyond 3 minutes).

### Default Telemetry Weights

- `w1 = 0.35` (Click-through indicator)
- `w2 = 0.40` (Normalized dwell duration)
- `w3 = 0.25` (Scroll depth percentage)
- `w1 + w2 + w3 = 1.0`

### Temporal Recency Decay

To reflect shifting user interests over time, historical interaction affinity scores undergo exponential decay:

`w_decay(delta_t) = exp(-lambda * delta_t)`

Where `delta_t` is the elapsed time in days, `lambda = ln(2) / half_life_days`, and `half_life_days = 7.0` days by default.

---

## Hybrid Blending and Scoring Architecture

The recommendation engine blends latent collaborative signals and semantic content representations using dynamic weights:

`Score = 0.60 * CF_score + 0.30 * Content_score + 0.10 * Category_boost`

### Scoring Components

1. **Two-Tower Neural Collaborative Filtering (`CF_score`):**
   - The user tower maps `user_id` to a 64-dimensional normalized latent vector `u`.
   - The item tower maps `article_id` to a 64-dimensional normalized latent vector `v`.
   - Prediction is calculated via dot product normalized to `[0.0, 1.0]`:
     `CF_score = (dot(u, v) + 1.0) / 2.0`

2. **Dense Semantic Content Matching (`Content_score`):**
   - Articles are encoded into 384-dimensional dense semantic vectors using `all-MiniLM-L6-v2`.
   - The user semantic profile is computed as the affinity-weighted centroid of recently consumed article vectors.
   - Prediction is calculated via cosine similarity normalized to `[0.0, 1.0]`:
     `Content_score = (cosine_similarity(user_profile, article_vector) + 1.0) / 2.0`

3. **Category Affinity Boost (`Category_boost`):**
   - Aggregated preference score for the article category based on the user historical engagement ratio.

### Cold-Start and Dynamic Routing Strategy

- **Known Collaborative Users:** Full hybrid scoring (`alpha = 0.60`, `beta = 0.10`).
- **Warm-Start Users (New / 1 to 4 reads):** Content-based semantic matching using real-time session history (`alpha = 0.0`).
- **Complete Cold-Start (0 reads):** Category-diverse popularity fallback with real-time impression deduplication.

---

## Benchmark Evaluation Results

Evaluated across 396 active users on held-out interaction test sets against the standard Popularity baseline:

| Model | Recall@5 | Recall@10 | NDCG@10 | MRR | Lift vs. Baseline (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Popularity Baseline | 0.0314 | 0.0630 | 0.0462 | 0.0693 | +0.0% |
| Content-Based Only | 0.0566 | 0.1219 | 0.0850 | 0.1290 | +93.6% |
| CF Two-Tower Only | 0.0549 | 0.1116 | 0.0816 | 0.1297 | +77.2% |
| Hybrid Engine (Full) | 0.0680 | 0.1272 | 0.0944 | 0.1495 | +102.0% |

### Key Benchmark Insights

- The **Hybrid Engine achieves a +102.0% lift in Recall@10** and a **+104.3% lift in NDCG@10** over the Popularity Baseline.
- The hybrid system outperforms both individual standalone models (`+14.0%` over CF Two-Tower alone and `+4.3%` over Content-Based alone on Recall@10).
- Latent collaborative filtering discovers serendipitous content, while semantic embeddings maintain precision for niche topics and warm-start users.

---

## Repository Layout

```text
content_rec_engine/
|-- data/
|   |-- download_data.py          # MIND dataset fetcher and synthetic telemetry generator
|   |-- schema.sql                # SQL DDL for SQLite/PostgreSQL feature store
|   `-- content_rec.db            # SQLite feature store database
|-- src/
|   |-- __init__.py
|   |-- database.py               # SQLAlchemy ORM connection layer and feature store
|   |-- telemetry_processor.py    # Multi-modal implicit signal weighting and normalization
|   |-- embeddings.py             # all-MiniLM-L6-v2 dense 384d semantic vector extractor
|   |-- models/
|   |   |-- __init__.py
|   |   |-- collaborative.py      # PyTorch Two-Tower neural matrix factorization
|   |   |-- content_based.py      # Cosine similarity on dense semantic vectors
|   |   `-- hybrid_engine.py      # Blended ranker with cold-start routing
|   |-- train.py                  # Training pipeline with checkpointing
|   |-- evaluate.py               # Quantitative benchmark module (Recall@K, NDCG@K, MRR)
|   `-- api/
|       |-- __init__.py
|       `-- app.py                # Low-latency FastAPI inference service
|-- tests/
|   |-- test_pipeline.py          # Unit tests for scoring, database, and models
|   `-- test_api.py               # Integration tests for FastAPI endpoints
|-- notebooks/
|   `-- model_benchmarks.ipynb    # Visual EDA and benchmark curves
|-- Dockerfile                    # Containerization for FastAPI service
|-- requirements.txt              # Pinned dependencies
`-- README.md                     # Documentation
```

---

## Quickstart Guide

### 1. Setup Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Download or Generate Data

Download the MIND dataset or generate synthetic news and telemetry:

```bash
python data/download_data.py --force-synthetic
```

### 3. Train Models and Populate Feature Store

Execute the end-to-end training pipeline. This computes semantic embeddings, ingests telemetry into SQLite, and trains the PyTorch Two-Tower model:

```bash
python src/train.py --force-synthetic --epochs 12
```

### 4. Run Quantitative Evaluation

Benchmark all models on held-out user interactions:

```bash
python src/evaluate.py
```

### 5. Run Unit and Integration Tests

Run the full pytest suite (14 tests covering telemetry, database, models, and API endpoints):

```bash
pytest -v tests/
```

### 6. Launch FastAPI Server

Start the FastAPI inference and telemetry service locally:

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

Interactive API documentation (Swagger UI) is available at `http://localhost:8000/docs`.

---

## API Endpoints Summary

### 1. Health Check (`GET /health`)

Check service availability, database connection, and model readiness.

```bash
curl -X GET "http://localhost:8000/health"
```

Response:

```json
{
  "status": "healthy",
  "timestamp": "2026-10-06T20:55:00.123456",
  "database_connected": true,
  "cf_model_loaded": true,
  "content_model_loaded": true,
  "articles_count": 300
}
```

### 2. Ingest Telemetry (`POST /interaction`)

Ingest real-time user interaction telemetry, calculate normalized composite affinity, and update user profiles.

```bash
curl -X POST "http://localhost:8000/interaction" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "U00042",
    "article_id": "N00015",
    "clicked": 1,
    "dwell_time": 92.5,
    "scroll_depth": 0.88
  }'
```

Response:

```json
{
  "status": "success",
  "interaction_id": 26125,
  "user_id": "U00042",
  "article_id": "N00015",
  "computed_affinity_score": 0.8252,
  "message": "Telemetry event ingested and affinity formulated successfully."
}
```

### 3. Get Recommendations (`GET /recommend/{user_id}`)

Retrieve personalized Top-K ranked recommendations with transparent explanation factors.

Query Parameters:
- `k` (optional, default: 10): Number of recommendations to retrieve.
- `filter_consumed` (optional, default: true): Filter out previously consumed articles.

```bash
curl -X GET "http://localhost:8000/recommend/U00042?k=5"
```

Response:

```json
{
  "user_id": "U00042",
  "recommendation_mode": "hybrid_two_tower_semantic",
  "count": 5,
  "latency_ms": 3.82,
  "recommendations": [
    {
      "article_id": "N00088",
      "title": "Chiefs Secure Dramatic Overtime Victory in AFC Championship",
      "category": "sports",
      "subcategory": "football_nfl",
      "url": "https://assets.msn.com/en-us/news/sports/N00088",
      "score": 0.8912,
      "cf_score": 0.8841,
      "content_score": 0.8983,
      "explanation": "Aligned with your high affinity for Sports stories."
    }
  ]
}
```

---

## Docker Deployment

Build and run the containerized FastAPI service:

```bash
docker build -t content-rec-engine:latest .
docker run -p 8000:8000 content-rec-engine:latest
```
