# Personalized Content Recommendation & Affinity Engine

An end-to-end, production-grade recommendation and implicit user affinity engine. The system integrates **PyTorch Neural Collaborative Filtering (Two-Tower Architecture)** with **Sentence-Transformers Semantic Embeddings (`all-MiniLM-L6-v2`)**, powered by a continuous **Implicit Interaction Telemetry Pipeline** (click-through, log-saturated dwell time, and viewport scroll depth) and a **Relational Feature Store**.

---

## 1. System Architecture

```mermaid
flowchart TD
    subgraph Client ["Client / Reading App"]
        Reader["Reader Session"]
        Telemetry["Implicit Telemetry Events<br/>(Click, Dwell Sec, Scroll %)"]
    end

    subgraph Ingestion ["Ingestion & Feature Store Layer"]
        API["FastAPI Inference Service<br/>(:8000)"]
        Processor["TelemetryProcessor<br/>Composite Affinity Engine"]
        DB[("Relational Feature Store<br/>SQLite / PostgreSQL")]
    end

    subgraph Models ["Hybrid Modeling & Embedding Layer"]
        CF["PyTorch Two-Tower CF<br/>(Latent Matrix Factorization)"]
        Content["Content-Based Engine<br/>(all-MiniLM-L6-v2 384d Dense Vectors)"]
        Hybrid["HybridRecommendationEngine<br/>Blended Ranker & Fallback Router"]
    end

    Reader -->|Logs Interaction| Telemetry
    Telemetry -->|POST /interaction| API
    API --> Processor
    Processor -->|Target Affinity in 0, 1| DB
    
    Reader -->|GET /recommend/{user_id}| API
    API --> Hybrid
    Hybrid -->|User Vector Dot-Product| CF
    Hybrid -->|Cosine Similarity| Content
    Hybrid -->|History & Catalog Query| DB
    Hybrid -->|Ranked Top-K with Explanations| API
```

---

## 2. Mathematical Formulation of Composite Affinity Score

Traditional recommenders treat user interest as a binary click ($y \in \{0, 1\}$). In modern content feeds, binary clicks suffer from clickbait noise and bounce distortion. This engine formulates a continuous affinity target score $A \in [0.0, 1.0]$ by fusing explicit and implicit engagement telemetry:

$$A = w_1 \cdot C + w_2 \cdot D_{\text{norm}} + w_3 \cdot S$$

Where:
- $C \in \{0, 1\}$ is the binary click-through indicator.
- $S \in [0.0, 1.0]$ is the continuous viewport vertical scroll depth percentage.
- $D_{\text{norm}} \in [0.0, 1.0]$ is the normalized reading dwell duration with log-scaled saturation:

$$D_{\text{norm}}(t) = \frac{\ln(1 + \min(t, \tau))}{\ln(1 + \tau)}$$

Here, $\tau = 180\text{s}$ represents reading saturation (diminishing marginal interest beyond 3 minutes). The default normalized weight allocation is:
$$w_1 = 0.35, \quad w_2 = 0.40, \quad w_3 = 0.25 \quad \left(\sum w_i = 1.0\right)$$

### Temporal Recency Decay
For past interactions occurring $\Delta t$ days ago, affinity weights are decayed exponentially:
$$w_{\text{decay}}(\Delta t) = \exp(-\lambda \Delta t) \quad \text{where } \lambda = \frac{\ln(2)}{t_{\text{half-life}}}$$

---

## 3. Hybrid Blending & Re-ranking Architecture

The engine blends collaborative filtering and semantic content similarity with dynamic routing:

$$S_{\text{hybrid}}(u, i) = \alpha \cdot S_{\text{CF}}(u, i) + (1 - \alpha) \cdot S_{\text{content}}(u, i) + \beta \cdot S_{\text{category}}(u, i)$$

1. Two-Tower Neural CF (S_{CF}):
$u = f_{\text{user}}(\text{user_id}) \in \mathbb{R}^{64}$ (L2 normalized)
$v = f_{\text{item}}(\text{item_id}) \in \mathbb{R}^{64}$ (L2 normalized)
Prediction: S_{CF} = \frac{\langle u, v \rangle + 1}{2} \in [0.0, 1.0]
```[cite: 9]

Notice `(S_{CF})` in the header, and `S_{CF} = ...` without bounding `$`: GitHub's KaTeX engine flags this as `'_' allowed only in math mode`[cite: 9].

#### Fix
Replace the entire **"3. Hybrid Blending & Re-ranking Architecture"** section in `README.md` with clean GitHub-compatible Markdown:

```markdown
## 3. Hybrid Blending & Re-ranking Architecture

The engine blends collaborative filtering and semantic content similarity with dynamic routing:

$$S_{\text{hybrid}}(u, i) = \alpha \cdot S_{\text{CF}}(u, i) + (1 - \alpha) \cdot S_{\text{content}}(u, i) + \beta \cdot S_{\text{category}}(u, i)$$

### 1. Two-Tower Neural CF ($S_{\text{CF}}$):
* **User Tower:** $\mathbf{u} = f_{\text{user}}(\text{user\_id}) \in \mathbb{R}^{64}$ (L2 normalized)
* **Item Tower:** $\mathbf{v} = f_{\text{item}}(\text{item\_id}) \in \mathbb{R}^{64}$ (L2 normalized)
* **Prediction:** $S_{\text{CF}} = \frac{\langle \mathbf{u}, \mathbf{v} \rangle + 1}{2} \in [0.0, 1.0]$

### 2. Dense Semantic Embeddings ($S_{\text{content}}$):
* Article representations extracted using `all-MiniLM-L6-v2` (384-dimensional dense vectors).
* **User Semantic Profile:** 
  $$\mathbf{u}_{\text{content}} = \frac{\sum_{j \in \mathcal{H}_u} A_j \mathbf{v}_j}{\sum_{j \in \mathcal{H}_u} A_j}$$
* **Similarity:** $S_{\text{content}} = \frac{\cos(\mathbf{u}_{\text{content}},\, \mathbf{v}_i) + 1}{2} \in [0.0, 1.0]$

### 3. Cold-Start Fallback Strategy:
* **Known CF Users:** Full hybrid scoring ($\alpha = 0.60, \beta = 0.10$).
* **Warm-Start Users (New / < 5 reads):** Content-based semantic matching using real-time session history ($\alpha = 0.0$).
* **Complete Cold-Start:** Category-diverse popularity fallback with real-time impression deduplication.

---

## 4. Quantitative Benchmark Evaluation

Evaluated across **396 active users** on held-out interaction test sets against the standard Popularity baseline:

| Model | Recall@5 | Recall@10 | NDCG@10 | MRR | Lift vs. Baseline (Recall@10) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Popularity Baseline** | 0.0314 | 0.0630 | 0.0462 | 0.0693 | — |
| **Content-Based Only** | 0.0566 | 0.1219 | 0.0850 | 0.1290 | +93.6% |
| **CF Two-Tower Only** | 0.0549 | 0.1116 | 0.0816 | 0.1297 | +77.2% |
| **Hybrid Engine (Full)** | **0.0680** | **0.1272** | **0.0944** | **0.1495** | **+102.0%** |

### Benchmark Takeaways
- The **Hybrid Engine achieves a +102.0% lift in Recall@10** and a **+104.3% lift in NDCG@10** over the Popularity Baseline.
- The hybrid model outperforms both standalone ablations (Content-Based and Neural CF) by effectively balancing latent collaborative discovery with semantic affinity.

---

## 5. Repository Layout

```
content_rec_engine/
├── data/
│   ├── download_data.py          # MIND dataset fetcher + high-fidelity synthetic fallback
│   ├── schema.sql                # SQL DDL for SQLite/PostgreSQL feature store
│   └── content_rec.db            # SQLite feature store database
├── src/
│   ├── __init__.py
│   ├── database.py               # SQLAlchemy ORM connection layer & feature store
│   ├── telemetry_processor.py    # Multi-modal implicit signal weighting & normalization
│   ├── embeddings.py             # all-MiniLM-L6-v2 dense 384d semantic vector extractor
│   ├── models/
│   │   ├── __init__.py
│   │   ├── collaborative.py      # PyTorch Two-Tower neural matrix factorization
│   │   ├── content_based.py      # Cosine similarity on dense semantic vectors
│   │   └── hybrid_engine.py      # Blended ranker with cold-start routing
│   ├── train.py                  # Training pipeline with checkpointing
│   ├── evaluate.py               # Quantitative benchmark module (Recall@K, NDCG@K, MRR)
│   └── api/
│       ├── __init__.py
│       └── app.py                # Low-latency FastAPI inference service
├── tests/
│   ├── test_pipeline.py          # Unit tests for scoring, database, and models
│   └── test_api.py               # Integration tests for FastAPI endpoints
├── notebooks/
│   └── model_benchmarks.ipynb    # Visual EDA, telemetry distributions, and benchmark curves
├── Dockerfile                    # Containerization for FastAPI service
├── requirements.txt              # Pinned production dependencies
└── README.md                     # Architecture, formulas, benchmark results, and guide
```

---

## 6. Local Execution Guide

### 6.1 Setup Virtual Environment
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 6.2 Data Ingestion & Model Training
Execute the complete end-to-end training pipeline. This fetches/synthesizes MIND telemetry, extracts transformer embeddings, writes to the feature store, and trains the Two-Tower PyTorch model:
```bash
python src/train.py --force-synthetic --epochs 12
```

### 6.3 Run Quantitative Evaluation
Benchmark all recommendation models on held-out user interactions:
```bash
python src/evaluate.py
```

### 6.4 Run Unit & Integration Tests
Execute the full pytest suite (14 tests covering telemetry, database, neural network, and API endpoints):
```bash
pytest -v tests/
```

### 6.5 Launch Inference API
Start the FastAPI service locally on port 8000:
```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger documentation is available at: `http://localhost:8000/docs`

---

## 7. API Specification & Examples

### Health Check
```bash
curl -X GET "http://localhost:8000/health"
```
**Response:**
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

### Log Real-Time Telemetry
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
**Response:**
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

### Get Personalized Recommendations
```bash
curl -X GET "http://localhost:8000/recommend/U00042?k=5"
```
**Response:**
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

## 8. Docker Deployment
```bash
docker build -t content-rec-engine:latest .
docker run -p 8000:8000 content-rec-engine:latest
```
