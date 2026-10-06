"""Implicit Telemetry Processor & Composite Affinity Engine.

Formulates continuous affinity scores from multi-modal implicit engagement signals:
- Binary Click-Through (CTR)
- Continuous Dwell Duration (seconds, log-saturated)
- Continuous Scroll Depth ([0.0, 1.0])

Formula:
    Affinity = w1 * click + w2 * normalized_dwell_time + w3 * scroll_depth
Bounded and normalized strictly to [0.0, 1.0].
"""

import math
import logging
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TelemetryProcessor:
    """Processes implicit interaction signals and calculates composite user affinities."""

    def __init__(
        self,
        weight_click: float = 0.35,
        weight_dwell: float = 0.40,
        weight_scroll: float = 0.25,
        dwell_saturation_sec: float = 180.0,
        dwell_scale_mode: str = "log",  # "log" or "linear"
        time_decay_half_life_days: float = 7.0
    ):
        """Initialize telemetry processor with configurable weights and saturation thresholds.

        Args:
            weight_click: Relative weight for explicit/implicit click event (w1).
            weight_dwell: Relative weight for reading dwell time (w2).
            weight_scroll: Relative weight for viewport scroll traversal (w3).
            dwell_saturation_sec: Reading time corresponding to 100% saturation.
            dwell_scale_mode: Scaling transformation: "log" (diminishing returns) or "linear".
            time_decay_half_life_days: Half-life for temporal recency decay.
        """
        total_w = weight_click + weight_dwell + weight_scroll
        if total_w <= 0:
            raise ValueError("Sum of telemetry weights must be strictly positive.")
            
        # Normalize weights so composite target is guaranteed in [0.0, 1.0]
        self.w1 = weight_click / total_w
        self.w2 = weight_dwell / total_w
        self.w3 = weight_scroll / total_w
        
        self.dwell_saturation_sec = max(1.0, dwell_saturation_sec)
        self.dwell_scale_mode = dwell_scale_mode
        self.decay_lambda = math.log(2) / max(0.1, time_decay_half_life_days)

    def normalize_dwell_time(self, dwell_seconds: Union[float, np.ndarray, pd.Series]) -> Union[float, np.ndarray, pd.Series]:
        """Normalize dwell duration in seconds to bounded [0.0, 1.0].
        
        Uses log-transform saturation by default to capture diminishing returns of long reads.
        """
        if isinstance(dwell_seconds, (pd.Series, np.ndarray)):
            clipped = np.clip(dwell_seconds, 0.0, None)
            if self.dwell_scale_mode == "log":
                denom = np.log1p(self.dwell_saturation_sec)
                norm = np.log1p(np.minimum(clipped, self.dwell_saturation_sec * 1.5)) / denom
                return np.clip(norm, 0.0, 1.0)
            else:
                return np.clip(clipped / self.dwell_saturation_sec, 0.0, 1.0)
        else:
            dwell = max(0.0, float(dwell_seconds))
            if self.dwell_scale_mode == "log":
                denom = math.log1p(self.dwell_saturation_sec)
                norm = math.log1p(min(dwell, self.dwell_saturation_sec * 1.5)) / denom
                return max(0.0, min(1.0, norm))
            else:
                return max(0.0, min(1.0, dwell / self.dwell_saturation_sec))

    def calculate_single_affinity(
        self,
        clicked: int,
        dwell_time: float,
        scroll_depth: float
    ) -> float:
        """Compute composite affinity for a single interaction event.
        
        Affinity = w1 * click + w2 * norm_dwell + w3 * scroll_depth
        Guaranteed to be within [0.0, 1.0].
        """
        c_val = 1.0 if clicked > 0 else 0.0
        d_val = self.normalize_dwell_time(dwell_time)
        s_val = max(0.0, min(1.0, float(scroll_depth)))
        
        affinity = (self.w1 * c_val) + (self.w2 * d_val) + (self.w3 * s_val)
        return float(np.clip(affinity, 0.0, 1.0))

    def process_telemetry_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process batch of telemetry rows, appending normalized dwell and composite affinity score."""
        df_out = df.copy()
        
        # Ensure requisite columns exist
        for col in ["clicked", "dwell_time", "scroll_depth"]:
            if col not in df_out.columns:
                raise KeyError(f"Expected column '{col}' in interaction telemetry dataframe.")
                
        # Fill missing values
        df_out["clicked"] = df_out["clicked"].fillna(0).astype(int)
        df_out["dwell_time"] = df_out["dwell_time"].fillna(0.0).astype(float)
        df_out["scroll_depth"] = df_out["scroll_depth"].fillna(0.0).astype(float).clip(0.0, 1.0)
        
        norm_dwell = self.normalize_dwell_time(df_out["dwell_time"].values)
        df_out["normalized_dwell"] = norm_dwell
        
        click_val = (df_out["clicked"] > 0).astype(float).values
        scroll_val = df_out["scroll_depth"].values
        
        raw_affinity = (self.w1 * click_val) + (self.w2 * norm_dwell) + (self.w3 * scroll_val)
        df_out["affinity_score"] = np.clip(raw_affinity, 0.0, 1.0).round(4)
        
        return df_out

    def compute_temporal_decay(self, event_time: datetime, reference_time: Optional[datetime] = None) -> float:
        """Compute exponential decay factor based on recency."""
        ref = reference_time or datetime.utcnow()
        delta_days = max(0.0, (ref - event_time).total_seconds() / 86400.0)
        return float(math.exp(-self.decay_lambda * delta_days))

    def build_user_affinity_profiles(
        self,
        interactions_df: pd.DataFrame,
        reference_time: Optional[datetime] = None
    ) -> Dict[str, Dict]:
        """Aggregate granular interactions into user preference profiles.
        
        Computes category-level preference weights and engagement statistics.
        """
        ref_time = reference_time or datetime.utcnow()
        profiles = {}
        
        grouped = interactions_df.groupby("user_id")
        for user_id, group in grouped:
            total_events = len(group)
            total_clicks = int(group["clicked"].sum())
            avg_dwell = float(group["dwell_time"].mean())
            avg_scroll = float(group["scroll_depth"].mean())
            
            # Category level affinities
            cat_scores: Dict[str, float] = {}
            has_categories = "category" in group.columns
            
            for _, row in group.iterrows():
                aff = float(row.get("affinity_score", 0.0))
                # Temporal decay if timestamp is present
                ts_str = row.get("timestamp")
                decay = 1.0
                if ts_str:
                    try:
                        ts = datetime.fromisoformat(str(ts_str))
                        decay = self.compute_temporal_decay(ts, ref_time)
                    except Exception:
                        decay = 1.0
                        
                weighted_aff = aff * decay
                if has_categories and pd.notna(row.get("category")):
                    cat = str(row["category"])
                    cat_scores[cat] = cat_scores.get(cat, 0.0) + weighted_aff

            # Normalize category distribution
            total_cat_sum = sum(cat_scores.values())
            if total_cat_sum > 0:
                normalized_cats = {k: round(v / total_cat_sum, 4) for k, v in cat_scores.items()}
            else:
                normalized_cats = {}
                
            profiles[user_id] = {
                "user_id": user_id,
                "total_interactions": total_events,
                "total_clicks": total_clicks,
                "avg_dwell_time": round(avg_dwell, 2),
                "avg_scroll_depth": round(avg_scroll, 3),
                "category_affinities": normalized_cats,
                "updated_at": ref_time.isoformat()
            }
            
        return profiles
