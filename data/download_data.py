"""MIND (Microsoft News Dataset) Ingestion & Synthetic Telemetry Generator.

Downloads public MIND-small dataset or generates a high-fidelity synthetic counterpart
matching the exact schema with simulated continuous implicit telemetry features
(dwell time, scroll depth) reflecting realistic news consumption patterns.
"""

import os
import io
import sys
import zipfile
import logging
import argparse
import random
from datetime import datetime, timedelta
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MIND_SMALL_TRAIN_URL = "https://mind201910.blob.core.windows.net/mind/MINDsmall_train.zip"
MIND_SMALL_DEV_URL = "https://mind201910.blob.core.windows.net/mind/MINDsmall_dev.zip"

CATEGORIES = {
    "sports": [
        ("football_nfl", "Super Bowl predictions, quarterback injuries, and draft rankings."),
        ("basketball_nba", "NBA playoffs race, triple-double records, and MVP candidate analysis."),
        ("soccer_epl", "Premier league title race, Champions League fixtures, and transfer rumors."),
        ("baseball_mlb", "World series recap, home run leaderboards, and pitching rotations.")
    ],
    "news": [
        ("worldnews", "Global diplomatic summits, geopolitical negotiations, and climate pacts."),
        ("nationalnews", "Congressional budget debates, infrastructure spending, and federal policies."),
        ("crime", "High-profile judicial hearings, investigation findings, and appellate decisions.")
    ],
    "finance": [
        ("markets", "Wall Street rallies, S&P 500 records, Treasury yields, and Federal Reserve rates."),
        ("personalfinance", "Roth IRA strategies, mortgage interest rate cuts, and high-yield savings."),
        ("realestate", "Commercial real estate distress, residential home inventory, and median pricing.")
    ],
    "technology": [
        ("artificialintelligence", "Generative AI foundation models, GPU clusters, and neural recommendation engines."),
        ("gadgets", "Smartphone battery innovations, flagship tablet comparisons, and wearable tech."),
        ("cybersecurity", "Zero-day vulnerabilities, ransomware defenses, and encrypted messaging standards.")
    ],
    "entertainment": [
        ("movies", "Box office weekend earnings, Academy Award predictions, and director interviews."),
        ("music", "Stadium tour announcements, streaming chart records, and album reviews."),
        ("celebrity", "Red carpet fashion reviews, celebrity philanthropy, and film festival premieres.")
    ],
    "lifestyle": [
        ("health_fitness", "Cardiovascular longevity routines, resistance training splits, and nutrition advice."),
        ("food_drink", "Artisanal espresso brewing, fermentation guides, and Michelin star culinary techniques."),
        ("travel", "Alpine backpacking itineraries, low-season flight deals, and remote eco-resorts.")
    ]
}

SAMPLE_TITLES = {
    "sports": [
        "Chiefs Secure Dramatic Overtime Victory in AFC Championship Thriller",
        "Lakers Surge Ahead Behind 40-Point Masterclass in Western Conference Clash",
        "Manchester City Retains Premier League Summit with Stoppage-Time Screamer",
        "Record-Breaking Strikeout Sequence Ignites Franchise Playoff Hopes",
        "Top Draft Prospect Declares for Draft Following Undefeated College Season"
    ],
    "news": [
        "Bipartisan Coalition Passes Landmark Clean Energy Infrastructure Framework",
        "Global Summit Concludes with Historic Accord on International Maritime Waters",
        "Supreme Court Delivers Unanimous Ruling on Digital Privacy and Device Encryption",
        "Severe Winter Storm Prompts Widespread Emergency Preparedness Across Midwest",
        "Mayor Unveils Comprehensive Downtown Revitalization and Light-Rail Transit Plan"
    ],
    "finance": [
        "Federal Reserve Signals Potential Interest Rate Pivot Amid Cooling Inflation",
        "Tech Heavyweights Propel S&P 500 to Fresh Record Highs in Resurgent Quarter",
        "Mortgage Rates Retreat to Six-Month Low, Sparking Surge in Home Loan Applications",
        "Venture Capital Investments Pivot Toward Hardware and Sovereign AI Startups",
        "Quarterly Corporate Earnings Exceed Projections as Consumer Spending Holds Resilient"
    ],
    "technology": [
        "Researchers Unveil Lightweight Neural Architecture for Real-Time On-Device Inference",
        "Next-Generation Quantum Processor Demonstrates Error Mitigation Milestone",
        "Major Operating System Update Introduces Zero-Trust Architecture and Passkeys",
        "High-Throughput Distributed Database Benchmarks Breakthrough in Transaction Latency",
        "Open-Source AI Consortium Releases Scalable Multimodal Reasoning Framework"
    ],
    "entertainment": [
        "Critically Acclaimed Sci-Fi Epic Dominates Global Box Office with Record Opening",
        "Indie Film Sensation Sweeps Festival Awards, Eyeing Major Autumn Theatrical Run",
        "Iconic Rock Legends Announce 50-City Global Farewell Stadium Tour",
        "Streaming Platform Renews Multi-Award Winning Dystopian Drama for Season Three",
        "Behind-the-Scenes Retrospective: How Master Directors Crafted Cinema's Greatest Shots"
    ],
    "lifestyle": [
        "Cardiologists Highlight 20-Minute Daily Walking Routine for Longevity Benefits",
        "The Science of Sourdough: Microbiome Secrets Behind Ancient Fermentation",
        "Hidden Mediterranean Coastal Villages That Escape the Peak Summer Crowds",
        "Ergonomic Workspace Layouts That Boost Cognitive Focus and Reduce Fatigue",
        "Mastering High-Altitude Hiking: Essential Gear, Hydration, and Acclimatization Tips"
    ]
}


def download_mind_dataset(dest_dir: str, dataset_url: str = MIND_SMALL_TRAIN_URL, timeout: int = 15) -> bool:
    """Attempt downloading and unpacking official MIND zip archive."""
    os.makedirs(dest_dir, exist_ok=True)
    behaviors_path = os.path.join(dest_dir, "behaviors.tsv")
    news_path = os.path.join(dest_dir, "news.tsv")

    if os.path.exists(behaviors_path) and os.path.exists(news_path):
        logger.info(f"Target files already exist at {dest_dir}. Skipping download.")
        return True

    logger.info(f"Attempting download from {dataset_url} (timeout: {timeout}s)...")
    try:
        response = requests.get(dataset_url, stream=True, timeout=timeout)
        if response.status_code == 200:
            logger.info("Download succeeded. Extracting archive...")
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                z.extractall(dest_dir)
            logger.info(f"Extracted MIND dataset to {dest_dir}")
            return True
        else:
            logger.warning(f"Download returned HTTP {response.status_code}.")
            return False
    except Exception as e:
        logger.warning(f"Download encountered network error/timeout: {e}")
        return False


def generate_synthetic_news(num_articles: int = 300, seed: int = 42) -> pd.DataFrame:
    """Generate realistic news dataframe matching MIND news.tsv schema.
    
    Columns: News ID, Category, SubCategory, Title, Abstract, URL, Title Entities, Abstract Entities
    """
    random.seed(seed)
    np.random.seed(seed)
    
    news_records = []
    category_keys = list(CATEGORIES.keys())
    
    for idx in range(1, num_articles + 1):
        news_id = f"N{idx:05d}"
        category = random.choice(category_keys)
        subcat, subcat_desc = random.choice(CATEGORIES[category])
        base_title = random.choice(SAMPLE_TITLES[category])
        
        # Add slight variation to ensure unique titles
        modifier_prefixes = ["Exclusive:", "Analysis:", "Special Report:", "Deep Dive:", "Update:", "Spotlight:"]
        prefix = random.choice(modifier_prefixes) if random.random() < 0.4 else ""
        title = f"{prefix} {base_title}".strip()
        
        abstract = (
            f"Comprehensive coverage regarding {subcat.replace('_', ' ')}. "
            f"{subcat_desc} Insights and detailed reporting on recent developments in {category}."
        )
        url = f"https://assets.msn.com/en-us/news/{category}/{news_id}"
        title_entities = "[]"
        abstract_entities = "[]"
        
        news_records.append({
            "news_id": news_id,
            "category": category,
            "subcategory": subcat,
            "title": title,
            "abstract": abstract,
            "url": url,
            "title_entities": title_entities,
            "abstract_entities": abstract_entities
        })
        
    df_news = pd.DataFrame(news_records)
    return df_news


def generate_synthetic_behaviors(
    news_df: pd.DataFrame,
    num_users: int = 400,
    impressions_per_user: int = 10,
    seed: int = 42
) -> pd.DataFrame:
    """Generate user behavior logs matching MIND behaviors.tsv schema.
    
    Creates distinct user persona clusters (e.g. sports fan, tech investor, generalist)
    to reflect authentic collaborative filtering and content preference signals.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    user_personas = ["sports_fan", "tech_finance", "lifestyle_culture", "general_news"]
    article_by_cat = {cat: news_df[news_df["category"] == cat]["news_id"].tolist() for cat in CATEGORIES.keys()}
    all_news_ids = news_df["news_id"].tolist()
    
    behavior_records = []
    impression_counter = 1
    base_time = datetime(2025, 11, 10, 8, 0, 0)
    
    for u_idx in range(1, num_users + 1):
        user_id = f"U{u_idx:05d}"
        persona = random.choice(user_personas)
        
        if persona == "sports_fan":
            preferred_cats = ["sports"]
            secondary_cats = ["news", "lifestyle"]
        elif persona == "tech_finance":
            preferred_cats = ["technology", "finance"]
            secondary_cats = ["news"]
        elif persona == "lifestyle_culture":
            preferred_cats = ["lifestyle", "entertainment"]
            secondary_cats = ["news"]
        else:
            preferred_cats = ["news", "finance"]
            secondary_cats = ["sports", "technology", "entertainment", "lifestyle"]
            
        # Generate browsing history
        history_pool = []
        for cat in preferred_cats:
            history_pool.extend(article_by_cat.get(cat, []))
        
        history_size = random.randint(4, 15)
        user_history = random.sample(history_pool, min(history_size, len(history_pool)))
        history_str = " ".join(user_history)
        
        # Generate impressions sessions
        for sess in range(impressions_per_user):
            sess_time = base_time + timedelta(
                days=random.randint(0, 5),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59)
            )
            time_str = sess_time.strftime("%m/%d/%Y %I:%M:%S %p")
            
            # Select candidate articles for this impression session
            session_candidates = []
            
            # 60% probability of showing preferred category articles
            preferred_candidates = []
            for cat in preferred_cats:
                preferred_candidates.extend(article_by_cat.get(cat, []))
            
            non_preferred_candidates = [n for n in all_news_ids if n not in preferred_candidates]
            
            num_preferred = random.randint(2, 4)
            num_other = random.randint(2, 5)
            
            sampled_pref = random.sample(preferred_candidates, min(num_preferred, len(preferred_candidates)))
            sampled_other = random.sample(non_preferred_candidates, min(num_other, len(non_preferred_candidates)))
            
            session_pool = sampled_pref + sampled_other
            random.shuffle(session_pool)
            
            impression_strings = []
            for art_id in session_pool:
                art_cat = news_df.loc[news_df["news_id"] == art_id, "category"].values[0]
                
                # Higher click probability if article matches user preferred persona
                if art_cat in preferred_cats:
                    p_click = 0.55
                elif art_cat in secondary_cats:
                    p_click = 0.20
                else:
                    p_click = 0.05
                    
                clicked = 1 if random.random() < p_click else 0
                impression_strings.append(f"{art_id}-{clicked}")
                
            impressions_str = " ".join(impression_strings)
            
            behavior_records.append({
                "impression_id": impression_counter,
                "user_id": user_id,
                "time": time_str,
                "history": history_str,
                "impressions": impressions_str
            })
            impression_counter += 1
            
    return pd.DataFrame(behavior_records)


def simulate_interaction_telemetry(behaviors_df: pd.DataFrame, news_df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Expand impression strings into granular interaction telemetry events.
    
    Simulates continuous implicit signals:
    - dwell_time (seconds): reading duration following log-normal distributions
    - scroll_depth (0.0 to 1.0): viewport vertical traversal percentage
    """
    random.seed(seed)
    np.random.seed(seed)
    
    category_map = dict(zip(news_df["news_id"], news_df["category"]))
    records = []
    
    for _, row in behaviors_df.iterrows():
        user_id = row["user_id"]
        sess_time = row["time"]
        try:
            ts = datetime.strptime(sess_time, "%m/%d/%Y %I:%M:%S %p")
        except Exception:
            ts = datetime.utcnow()
            
        imp_tokens = str(row["impressions"]).strip().split()
        for token in imp_tokens:
            if "-" not in token:
                continue
            art_id, click_flag = token.split("-", 1)
            clicked = int(click_flag)
            
            if clicked == 0:
                # User glanced past or skipped impression
                dwell_time = round(float(np.random.uniform(0.5, 3.5)), 2)
                scroll_depth = round(float(np.clip(np.random.beta(1.5, 8.0), 0.02, 0.25)), 3)
            else:
                # User engaged with article
                engagement_type = np.random.choice(["bounce", "skim", "deep_read"], p=[0.15, 0.35, 0.50])
                if engagement_type == "bounce":
                    dwell_time = round(float(np.random.uniform(4.0, 12.0)), 2)
                    scroll_depth = round(float(np.random.uniform(0.15, 0.40)), 3)
                elif engagement_type == "skim":
                    dwell_time = round(float(np.clip(np.random.lognormal(mean=3.4, sigma=0.4), 15.0, 65.0)), 2)
                    scroll_depth = round(float(np.random.uniform(0.40, 0.75)), 3)
                else:  # deep_read
                    dwell_time = round(float(np.clip(np.random.lognormal(mean=4.4, sigma=0.5), 50.0, 300.0)), 2)
                    scroll_depth = round(float(np.clip(np.random.beta(6.0, 1.2), 0.75, 1.0)), 3)
                    
            records.append({
                "user_id": user_id,
                "article_id": art_id,
                "category": category_map.get(art_id, "unknown"),
                "clicked": clicked,
                "dwell_time": dwell_time,
                "scroll_depth": scroll_depth,
                "timestamp": ts.isoformat()
            })
            
    df_telemetry = pd.DataFrame(records)
    logger.info(f"Synthesized {len(df_telemetry)} telemetry records across {len(behaviors_df)} impressions.")
    return df_telemetry


def ensure_dataset(data_dir: str, fallback_only: bool = False, num_articles: int = 300, num_users: int = 400) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Ensure news.tsv and behaviors.tsv are present in data_dir, downloading or generating as needed."""
    os.makedirs(data_dir, exist_ok=True)
    behaviors_path = os.path.join(data_dir, "behaviors.tsv")
    news_path = os.path.join(data_dir, "news.tsv")
    
    if os.path.exists(behaviors_path) and os.path.exists(news_path):
        logger.info(f"Loading existing datasets from {data_dir}...")
        news_cols = ["news_id", "category", "subcategory", "title", "abstract", "url", "title_entities", "abstract_entities"]
        df_news = pd.read_csv(news_path, sep="\t", header=None, names=news_cols, quoting=3, on_bad_lines="skip")
        behaviors_cols = ["impression_id", "user_id", "time", "history", "impressions"]
        df_behaviors = pd.read_csv(behaviors_path, sep="\t", header=None, names=behaviors_cols, quoting=3, on_bad_lines="skip")
        return df_news, df_behaviors

    download_success = False
    if not fallback_only:
        download_success = download_mind_dataset(data_dir, timeout=8)

    if download_success and os.path.exists(behaviors_path) and os.path.exists(news_path):
        news_cols = ["news_id", "category", "subcategory", "title", "abstract", "url", "title_entities", "abstract_entities"]
        df_news = pd.read_csv(news_path, sep="\t", header=None, names=news_cols, quoting=3, on_bad_lines="skip")
        behaviors_cols = ["impression_id", "user_id", "time", "history", "impressions"]
        df_behaviors = pd.read_csv(behaviors_path, sep="\t", header=None, names=behaviors_cols, quoting=3, on_bad_lines="skip")
        return df_news, df_behaviors

    logger.info("Generating synthetic MIND dataset matching exact production schemas...")
    df_news = generate_synthetic_news(num_articles=num_articles)
    df_behaviors = generate_synthetic_behaviors(df_news, num_users=num_users)
    
    # Save TSV matching MIND conventions (no header)
    df_news.to_csv(news_path, sep="\t", header=False, index=False)
    df_behaviors.to_csv(behaviors_path, sep="\t", header=False, index=False)
    logger.info(f"Generated and persisted {len(df_news)} news items and {len(df_behaviors)} impressions to {data_dir}")
    return df_news, df_behaviors


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch or generate MIND news dataset with implicit telemetry.")
    parser.add_argument("--output-dir", type=str, default="./data", help="Target directory for TSV files.")
    parser.add_argument("--force-synthetic", action="store_true", help="Bypass remote download and synthesize locally.")
    parser.add_argument("--num-articles", type=int, default=300, help="Number of synthetic articles.")
    parser.add_argument("--num-users", type=int, default=400, help="Number of synthetic users.")
    args = parser.parse_args()
    
    news_df, behav_df = ensure_dataset(
        data_dir=args.output_dir,
        fallback_only=args.force_synthetic,
        num_articles=args.num_articles,
        num_users=args.num_users
    )
    print(f"Dataset ready. News count: {len(news_df)}, Impressions count: {len(behav_df)}")
