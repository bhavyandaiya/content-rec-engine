# Complete Beginner's Architectural Guide & Execution Handbook
### *Personalized Content Recommendation & Affinity Engine*

If you do not want to open a single Python file, read this document. This guide explains:
1. **What modular programming is** and why this repository is organized the way it is.
2. **How every single file works inside**, what code lives inside each module, and how they communicate.
3. **How to run the entire system from your terminal** in under 2 minutes without touching any code.
4. **How to see recommendations and interact with the live API** using simple curl commands or a web browser.

---

## Table of Contents
1. [The "Mental Model": What is Module-Level Programming?](#1-the-mental-model-what-is-module-level-programming)
2. [Big Picture: How the System Works End-to-End](#2-big-picture-how-the-system-works-end-to-end)
3. [Module-by-Module Code Explanation (Without Opening Files)](#3-module-by-module-code-explanation)
   - [3.1 Data Layer (`data/download_data.py` & `data/schema.sql`)](#31-data-layer)
   - [3.2 Database Manager (`src/database.py`)](#32-database-manager)
   - [3.3 Telemetry Processor (`src/telemetry_processor.py`)](#33-telemetry-processor)
   - [3.4 Embedding Extractor (`src/embeddings.py`)](#34-embedding-extractor)
   - [3.5 Neural Collaborative Filtering (`src/models/collaborative.py`)](#35-neural-collaborative-filtering)
   - [3.6 Content-Based Filtering (`src/models/content_based.py`)](#36-content-based-filtering)
   - [3.7 Hybrid Decision Engine (`src/models/hybrid_engine.py`)](#37-hybrid-decision-engine)
   - [3.8 Training Orchestrator (`src/train.py`)](#38-training-orchestrator)
   - [3.9 Evaluation Benchmark (`src/evaluate.py`)](#39-evaluation-benchmark)
   - [3.10 Web API Server (`src/api/app.py`)](#310-web-api-server)
   - [3.11 Test Suite (`tests/`)](#311-test-suite)
4. [Zero-Touch Running Guide: Exact Terminal Commands](#4-zero-touch-running-guide)
5. [How to Test and Verify Everything](#5-how-to-test-and-verify-everything)

---

## 1. The "Mental Model": What is Module-Level Programming?

When beginners write Python, they often write one gigantic script called `script.py` with 2,000 lines. The script downloads data, defines functions, runs math, connects to a database, and runs a web server all in one place. 

Why is that bad?
- If one small part breaks (e.g., your database crashes), the whole script crashes.
- It is impossible to test small functions in isolation.
- You cannot reuse code without copying and pasting.

**Module-level programming** is like building a **professional restaurant kitchen**:
- The **Baker** only bakes bread (`embeddings.py`).
- The **Dishwasher** only cleans and organizes plates (`database.py`).
- The **Sous Chef** preps raw ingredients into neat bowls (`telemetry_processor.py`).
- The **Head Chef** assembles the final plate (`hybrid_engine.py`).
- The **Waiter** interacts with the customers outside (`api/app.py`).

Each Python file (`.py`) is called a **module**. A folder containing modules is called a **package** (indicated by an `__init__.py` file). Each module only does **one job**, and when it needs help, it uses `import` to ask another module for assistance.

---

## 2. Big Picture: How the System Works End-to-End

Here is the journey of data through the codebase:

```
[User Reads News on App]
       │
       ▼ (User scrolled 80%, stayed 90 seconds, clicked)
[src/api/app.py] ─── Sends raw numbers to ───► [src/telemetry_processor.py]
                                                       │
                                                       ▼ Calculates composite score: 0.89
                                              [src/database.py]
                                                       │ (Stores in SQLite)
                                                       ▼
[Candidate Articles] ◄── Evaluates ─── [src/models/hybrid_engine.py]
                                            │               │
                            Ask for CF Score│               │Ask for Semantic Score
                                            ▼               ▼
                       [collaborative.py]       [content_based.py]
                       (PyTorch Two-Tower)      (all-MiniLM-L6-v2)
                                            │               │
                                            └───────┬───────┘
                                                    ▼
                                            Blends the scores
                                                    ▼
                                      [Top-10 Articles Sent to User]
```

---

## 3. Module-by-Module Code Explanation

### 3.1 Data Layer

#### File: `content_rec_engine/data/schema.sql`
- **What it is**: The blueprints for the SQL database tables.
- **What it creates**:
  1. `articles`: Holds ID, category (e.g., sports, finance), title, abstract, URL, and precomputed semantic embeddings.
  2. `users`: Holds user IDs and timestamps of when they were first seen and last active.
  3. `interactions`: The raw telemetry ledger. Every time someone interacts with an article, a row is recorded containing `(user_id, article_id, clicked, dwell_time, scroll_depth, affinity_score, timestamp)`.
  4. `user_affinity_profiles`: Aggregated statistics for each user (e.g., total clicks, average read time, and category interest percentages like `{sports: 0.70, tech: 0.30}`).
- **Why it matters**: It ensures data is clean, indexed, and queryable in milliseconds.

#### File: `content_rec_engine/data/download_data.py`
- **What it is**: The data loader and synthetic generator.
- **What it does**:
  1. It tries to download Microsoft's public news dataset (MIND).
  2. If you are offline, or Microsoft's server is slow, it automatically runs an internal **generator** that creates identical realistic news articles across categories (`sports`, `finance`, `technology`, `entertainment`, etc.) and simulates user cohorts (e.g., sports fans, tech junkies).
  3. It simulates real human browsing behavior:
     - Non-clicks: 1-3 seconds glanced past, <25% scroll.
     - Clicks that bounced: 4-10 seconds read, 20% scroll.
     - Deep reads: 50-300 seconds read, 80-100% scroll traversal.
  4. It saves two files: `news.tsv` and `behaviors.tsv`.

---

### 3.2 Database Manager

#### File: `content_rec_engine/src/database.py`
- **What it is**: The bridge between Python code and SQLite/PostgreSQL.
- **Key Concepts Used**: **SQLAlchemy ORM** (Object Relational Mapping). Instead of writing raw SQL strings like `"SELECT * FROM articles"`, we write Python code like `session.query(ArticleModel).all()`.
- **Key Classes & Methods**:
  - `ArticleModel`, `UserModel`, `InteractionModel`, `UserAffinityProfileModel`: Python classes representing the database tables.
  - `DatabaseManager`:
    - `upsert_articles(list_of_articles)`: Saves or updates articles in bulk.
    - `log_interaction(user_id, article_id, clicked, dwell_time, scroll_depth, affinity_score)`: Appends an interaction row to the database.
    - `get_popular_articles(limit=10)`: Calculates the most popular articles on the platform (used when a new user visits for the very first time).
    - `get_user_interactions(user_id)`: Fetches a specific reader's past reading history.

---

### 3.3 Telemetry Processor

#### File: `content_rec_engine/src/telemetry_processor.py`
- **What it is**: The mathematical engine that converts user behavior into a single number between 0.0 and 1.0.
- **The Problem It Solves**: In old recommender systems, clicking an article was marked as `1`, and not clicking was `0`. But what if a reader clicked clickbait and left after 2 seconds? That should NOT count as high interest!
- **The Mathematical Formula**:
  $$\text{Affinity} = (w_1 \cdot \text{click}) + (w_2 \cdot \text{normalized\_dwell}) + (w_3 \cdot \text{scroll\_depth})$$
  - $w_1 = 0.35$ (Click weight)
  - $w_2 = 0.40$ (Dwell time weight)
  - $w_3 = 0.25$ (Scroll traversal weight)
- **Key Methods**:
  - `normalize_dwell_time(dwell_seconds)`: Uses logarithmic saturation ($\ln(1 + t) / \ln(1 + 180)$). Reading for 30 seconds gives a big boost; reading for 600 seconds instead of 180 seconds gives diminishing marginal returns.
  - `calculate_single_affinity(clicked, dwell_time, scroll_depth)`: Takes 3 numbers and returns a score strictly between 0.0 (zero interest) and 1.0 (maximum deep read).
  - `build_user_affinity_profiles(df)`: Aggregates all interactions of a reader to calculate what percent of their attention goes to Sports vs Finance vs Tech.

---

### 3.4 Embedding Extractor

#### File: `content_rec_engine/src/embeddings.py`
- **What it is**: The text understanding module.
- **How it works**:
  - It loads the Hugging Face transformer model **`all-MiniLM-L6-v2`** using `sentence-transformers`.
  - It takes an article's title, category, and abstract (e.g., *"Chiefs Secure Overtime Victory in AFC Championship"*) and converts it into a list of **384 numbers** (a dense vector).
  - In this 384-dimensional space, articles about American Football and articles about Soccer will naturally sit close to each other, while articles about Cryptocurrency sit far away.
  - Vectors are automatically normalized to unit length ($L_2 = 1.0$), meaning the cosine similarity between any two articles is calculated via a fast matrix dot product.

---

### 3.5 Neural Collaborative Filtering

#### File: `content_rec_engine/src/models/collaborative.py`
- **What it is**: A **PyTorch Two-Tower Deep Neural Network**.
- **The Concept**:
  - "Collaborative Filtering" means recommending articles based on what **other readers with similar tastes** enjoyed.
  - **Tower 1 (User Tower)**: Takes a `user_id` $\to$ looks up a learned embedding $\to$ passes it through a Multi-Layer Perceptron (Linear $\to$ ReLU $\to$ Dropout $\to$ Linear) $\to$ outputs a 64-dimensional user vector.
  - **Tower 2 (Item Tower)**: Takes an `article_id` $\to$ looks up a learned embedding $\to$ passes it through a Multi-Layer Perceptron $\to$ outputs a 64-dimensional article vector.
  - **Dot Product**: Multiplies the user vector by the article vector. If the user and article match, the score is high!
- **Key Methods**:
  - `forward(user_idx, item_idx)`: Calculates the predicted affinity score $[0.0, 1.0]$.
  - `predict_all_for_user(user_idx)`: Matrix multiplies the user's vector against all articles in the catalog in one shot.
  - `save_checkpoint(path)` & `load_checkpoint(path)`: Saves and reloads the trained PyTorch weights and vocabulary dictionary from disk.

---

### 3.6 Content-Based Filtering

#### File: `content_rec_engine/src/models/content_based.py`
- **What it is**: Recommender that finds articles semantically similar to what the user read recently.
- **How it works**:
  1. It inspects the articles the user read.
  2. It computes an **Average User Semantic Vector**:
     $$\vec{u} = \frac{\sum \text{Affinity}_i \cdot \vec{v}_i}{\sum \text{Affinity}_i}$$
  3. It then takes this user vector and compares it against every article in the catalog using cosine similarity.
  4. Articles that discuss similar topics to what the user has spent high dwell-time reading get the highest score.

---

### 3.7 Hybrid Decision Engine

#### File: `content_rec_engine/src/models/hybrid_engine.py`
- **What it is**: The master brain that blends all recommendations together.
- **Why a Hybrid?**:
  - Collaborative filtering alone fails when a brand new user arrives (**Cold-Start Problem**).
  - Content-based filtering alone can trap users in a bubble (only showing identical articles).
  - The hybrid engine intelligently blends both:
    $$\text{Score} = (\alpha \cdot \text{Score}_{\text{CF}}) + ((1 - \alpha) \cdot \text{Score}_{\text{Content}}) + (\beta \cdot \text{CategoryBoost})$$
- **Smart Fallback Hierarchy**:
  - If the user is completely new (0 history): Routes to **Popularity Recommender** across diverse categories.
  - If the user has read 1 or 2 articles today: Routes to **Content-Based Warm-Start**.
  - If the user is an established reader: Uses **Full Hybrid Blending** ($\alpha = 0.60$).
  - Filters out articles the user has already read so they never see duplicate content.
  - Generates clear, human-readable explanations (e.g., *"Semantically similar to your recent reads in Sports"*).

---

### 3.8 Training Orchestrator

#### File: `content_rec_engine/src/train.py`
- **What it is**: The end-to-end training pipeline.
- **What happens when you run it**:
  1. Ingests or generates the news dataset.
  2. Runs the `TelemetryProcessor` to formulate target affinity scores.
  3. Encodes all articles into 384-dimensional dense vectors using `all-MiniLM-L6-v2` and caches them to `checkpoints/content_embeddings.npz`.
  4. Bulk inserts all articles, users, telemetry interactions, and user profiles into the SQLite feature store `data/content_rec.db`.
  5. Trains the PyTorch `TwoTowerRecommender` neural network across 12 epochs using AdamW and MSE loss.
  6. Saves the best model checkpoint to `checkpoints/two_tower_best.pt`.

---

### 3.9 Evaluation Benchmark

#### File: `content_rec_engine/src/evaluate.py`
- **What it is**: The scientific proof and benchmarking tool.
- **What it does**:
  1. Takes the user interaction history and splits it into **training** (past) and **test** (future held-out reads).
  2. Evaluates four different algorithms on the exact same test users:
     - Baseline 1: **Popularity Baseline** (ranks articles by raw total clicks).
     - Baseline 2: **Content-Based Only** (semantic cosine similarity).
     - Baseline 3: **CF Two-Tower Only** (neural matrix factorization).
     - Champion: **Hybrid Engine (Full)** (the complete blended model).
  3. Computes industry-standard ranking metrics:
     - **Recall@K**: Did the user's favorite articles appear in the top K recommendations?
     - **NDCG@10**: Were the best articles placed at the very top of the list?
     - **MRR (Mean Reciprocal Rank)**: How quickly does the first relevant article appear?
  4. Prints a comparison table showing that the Hybrid Model delivers a **+102% lift in Recall@10** over the Popularity Baseline.

---

### 3.10 Web API Server

#### File: `content_rec_engine/src/api/app.py`
- **What it is**: A high-performance **FastAPI** web service.
- **What endpoints it provides**:
  1. `GET /health`: Checks if the database is connected, models are loaded into memory, and reports total article counts.
  2. `POST /interaction`: Allows web or mobile frontends to stream real-time telemetry (`clicked`, `dwell_time`, `scroll_depth`). It calculates the affinity score in real-time, persists it to SQL, and updates the user profile cache.
  3. `GET /recommend/{user_id}?k=10`: Returns the Top-10 personalized articles for that reader with full titles, URLs, scores, and explanations in under 30 milliseconds.
  4. Interactive Swagger documentation at `http://127.0.0.1:8000/docs`.

---

### 3.11 Test Suite

#### Files: `content_rec_engine/tests/test_pipeline.py` & `test_api.py`
- **What they are**: Automated unit and integration tests.
- **What they verify**:
  - Test that affinity scores are mathematically bounded between 0.0 and 1.0.
  - Test that a 120-second read scores higher than a 3-second bounce click.
  - Test database CRUD operations and foreign key relationships.
  - Test the PyTorch neural network forward pass and embedding shapes.
  - Test API endpoints for valid inputs, invalid inputs, and cold-start fallbacks.

---

## 4. Zero-Touch Running Guide

You do not need to open or edit any code. Run these exact shell commands in your terminal:

### Step 1: Open Terminal & Activate the Virtual Environment
Navigate to the project root directory and activate the existing virtual environment:
```bash
cd /Users/daiyabhavyan/Documents/Studies/content-recommendation-data-science
source .venv/bin/activate
```

*(Note: All packages like `torch`, `sentence-transformers`, `fastapi`, and `pandas` are already installed inside `.venv`)*

---

### Step 2: Run the Unit Tests (Verifies Everything is Working)
Run pytest to ensure all 14 tests pass:
```bash
pytest -v content_rec_engine/tests
```
**Expected Output:**
```
14 passed in ~4s (100% test success)
```

---

### Step 3: Run the Training Pipeline
To generate the dataset, calculate implicit affinities, extract transformer embeddings, and train the neural network:
```bash
python content_rec_engine/src/train.py --force-synthetic --epochs 12
```
**What happens behind the scenes:**
- 300 news articles and 26,000+ telemetry rows are created in `content_rec_engine/data/content_rec.db`.
- Semantic embeddings are saved to `content_rec_engine/checkpoints/content_embeddings.npz`.
- PyTorch Two-Tower model trains for 12 epochs and saves to `content_rec_engine/checkpoints/two_tower_best.pt`.

---

### Step 4: Run the Benchmark Evaluation
To see the model performance metrics and comparison table:
```bash
python content_rec_engine/src/evaluate.py --db-path content_rec_engine/data/content_rec.db --checkpoint-dir content_rec_engine/checkpoints
```
**Expected Output:**
```
================================================================================
                  TOP-K RECOMMENDATION BENCHMARK EVALUATION
================================================================================
| Model                |   Recall@5 |   Recall@10 |   NDCG@10 |    MRR | Lift vs. Baseline (%)   |
|:---------------------|-----------:|------------:|----------:|-------:|:------------------------|
| Popularity Baseline  |     0.0314 |      0.0630 |    0.0462 | 0.0693 | +0.0%                   |
| Content-Based Only   |     0.0566 |      0.1219 |    0.0850 | 0.1290 | +93.6%                  |
| CF Two-Tower Only    |     0.0549 |      0.1116 |    0.0816 | 0.1297 | +77.2%                  |
| Hybrid Engine (Full) |     0.0680 |      0.1272 |    0.0944 | 0.1495 | +102.0%                 |
================================================================================
```

---

### Step 5: Start the Production API Server
To launch the FastAPI service:
```bash
cd content_rec_engine
python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000 --reload
```
You will see:
```
INFO: Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

Leave this terminal running.

---

## 5. How to Test and Verify Everything

Open a **new terminal tab or window** to test the running API:

### 1. Check Server Health
```bash
curl -s http://127.0.0.1:8000/health
```
**Response:**
```json
{
  "status": "healthy",
  "database_connected": true,
  "cf_model_loaded": true,
  "content_model_loaded": true,
  "articles_count": 300
}
```

---

### 2. Get Recommendations for an Active User
Ask for 5 recommendations for user `U00001` (a sports fan):
```bash
curl -s "http://127.0.0.1:8000/recommend/U00001?k=5"
```
**Response:**
```json
{
  "user_id": "U00001",
  "recommendation_mode": "hybrid_two_tower_semantic",
  "count": 5,
  "latency_ms": 28.3,
  "recommendations": [
    {
      "article_id": "N00028",
      "title": "Lakers Surge Ahead Behind 40-Point Masterclass in Western Conference Clash",
      "category": "sports",
      "score": 0.7137,
      "explanation": "Semantically similar to your recent reads in Sports."
    }
  ]
}
```

---

### 3. Log a Real-Time Interaction (Watch the Score Update)
Simulate a user reading an article with 95 seconds dwell time and 85% scroll depth:
```bash
curl -s -X POST http://127.0.0.1:8000/interaction \
  -H "Content-Type: application/json" \
  -d '{"user_id": "U00001", "article_id": "N00010", "clicked": 1, "dwell_time": 95.0, "scroll_depth": 0.85}'
```
**Response:**
```json
{
  "status": "success",
  "interaction_id": 26125,
  "computed_affinity_score": 0.9137,
  "message": "Telemetry event ingested and affinity formulated successfully."
}
```

---

### 4. Test Cold-Start for a Brand-New User
Ask for recommendations for a reader who has never visited before (`GUEST_999`):
```bash
curl -s "http://127.0.0.1:8000/recommend/GUEST_999?k=3"
```
**Response:**
```json
{
  "user_id": "GUEST_999",
  "recommendation_mode": "cold_start_popularity",
  "count": 3,
  "recommendations": [
    {
      "article_id": "N00074",
      "title": "Top Draft Prospect Declares for Draft Following Undefeated College Season",
      "category": "sports",
      "score": 0.4115,
      "explanation": "Trending in Sports"
    }
  ]
}
```
Notice how it automatically falls back to `cold_start_popularity` with diverse trending topics without crashing!

---

### 5. Interactive Swagger Web UI
If you prefer a visual web interface over terminal commands, open your web browser and navigate to:
```
http://127.0.0.1:8000/docs
```
You will see interactive documentation where you can click **"Try it out"** on any endpoint and inspect responses directly.
