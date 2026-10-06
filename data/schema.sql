-- ============================================================================
-- Personalized Content Recommendation & Affinity Engine
-- Relational Feature Store Schema (SQLite / PostgreSQL Compatible)
-- ============================================================================

-- Table: articles
-- Stores news/content metadata and optional precomputed topic/semantic vectors
CREATE TABLE IF NOT EXISTS articles (
    article_id VARCHAR(64) PRIMARY KEY,
    category VARCHAR(64) NOT NULL,
    subcategory VARCHAR(64),
    title TEXT NOT NULL,
    abstract TEXT,
    url TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    embedding_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_articles_category ON articles(category);

-- Table: users
-- Stores user entities and registration/activity timestamps
CREATE TABLE IF NOT EXISTS users (
    user_id VARCHAR(64) PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table: interactions (Implicit Telemetry Store)
-- Captures granular session telemetry: click flag, dwell duration, and scroll depth
CREATE TABLE IF NOT EXISTS interactions (
    interaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id VARCHAR(64) NOT NULL,
    article_id VARCHAR(64) NOT NULL,
    clicked INTEGER NOT NULL DEFAULT 0,            -- 0 or 1 binary flag
    dwell_time REAL NOT NULL DEFAULT 0.0,          -- Read duration in seconds
    scroll_depth REAL NOT NULL DEFAULT 0.0,        -- Percentage in [0.0, 1.0]
    affinity_score REAL NOT NULL DEFAULT 0.0,      -- Composite weighted target score [0.0, 1.0]
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
    FOREIGN KEY (article_id) REFERENCES articles(article_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_interactions_user ON interactions(user_id);
CREATE INDEX IF NOT EXISTS idx_interactions_article ON interactions(article_id);
CREATE INDEX IF NOT EXISTS idx_interactions_timestamp ON interactions(timestamp);
CREATE INDEX IF NOT EXISTS idx_interactions_composite ON interactions(user_id, article_id);

-- Table: user_affinity_profiles
-- Stores real-time aggregated profile statistics and category affinities
CREATE TABLE IF NOT EXISTS user_affinity_profiles (
    user_id VARCHAR(64) PRIMARY KEY,
    total_interactions INTEGER DEFAULT 0,
    total_clicks INTEGER DEFAULT 0,
    avg_dwell_time REAL DEFAULT 0.0,
    avg_scroll_depth REAL DEFAULT 0.0,
    category_affinities_json TEXT,                -- Map of {category: affinity_sum}
    latent_vector_json TEXT,                      -- Serialized latent user embedding
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);
