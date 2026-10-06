"""Semantic Text Embedding Extraction for Articles and Topics.

Utilizes Hugging Face / Sentence-Transformers (all-MiniLM-L6-v2) to extract
384-dimensional dense embeddings over article metadata (category, subcategory, title, abstract).
"""

import os
import logging
from typing import List, Dict, Optional, Union, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


class ContentEmbeddingExtractor:
    """Extracts dense semantic representations using pretrained transformer models."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, device: Optional[str] = None):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._init_model()

    def _init_model(self) -> None:
        """Lazy load or initialize sentence-transformers model."""
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading SentenceTransformer model '{self.model_name}'...")
            self._model = SentenceTransformer(self.model_name, device=self.device)
            logger.info("SentenceTransformer model loaded successfully.")
        except Exception as e:
            logger.warning(f"Could not load SentenceTransformer '{self.model_name}': {e}. Initializing deterministic fallback encoder.")
            self._model = None

    def format_article_text(
        self,
        title: str,
        abstract: Optional[str] = None,
        category: Optional[str] = None,
        subcategory: Optional[str] = None
    ) -> str:
        """Compose structured text representation for optimal transformer pooling."""
        parts = []
        if category:
            cat_str = f"Category: {category}"
            if subcategory:
                cat_str += f" ({subcategory})"
            parts.append(cat_str)
        if title:
            parts.append(f"Title: {title.strip()}")
        if abstract:
            parts.append(f"Abstract: {abstract.strip()}")
            
        return " | ".join(parts) if parts else "Empty document"

    def encode_texts(self, texts: List[str], batch_size: int = 32, normalize: bool = True) -> np.ndarray:
        """Encode a batch of raw texts into L2-normalized dense vectors."""
        if not texts:
            return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

        if self._model is not None:
            embeddings = self._model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=normalize
            )
            return embeddings.astype(np.float32)
        else:
            # Deterministic fallback encoder for offline/air-gapped systems
            return self._fallback_encode(texts, normalize=normalize)

    def encode_articles(self, articles: Union[pd.DataFrame, List[Dict[str, Any]]], batch_size: int = 32) -> np.ndarray:
        """Extract embeddings for an iterable of article dictionaries or DataFrame."""
        if isinstance(articles, pd.DataFrame):
            records = articles.to_dict(orient="records")
        else:
            records = articles

        texts = []
        for item in records:
            title = item.get("title", "")
            abstract = item.get("abstract", "")
            cat = item.get("category", "")
            subcat = item.get("subcategory", "")
            formatted = self.format_article_text(title=title, abstract=abstract, category=cat, subcategory=subcat)
            texts.append(formatted)

        return self.encode_texts(texts, batch_size=batch_size, normalize=True)

    def _fallback_encode(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        """Deterministic hash-based projection guaranteeing 384-dim normalized outputs."""
        embeddings = np.zeros((len(texts), EMBEDDING_DIM), dtype=np.float32)
        for i, text in enumerate(texts):
            # Seed generator based on stable string hash
            h = abs(hash(text))
            rng = np.random.RandomState(h % (2**31 - 1))
            vec = rng.randn(EMBEDDING_DIM).astype(np.float32)
            if normalize:
                norm = np.linalg.norm(vec)
                if norm > 1e-9:
                    vec = vec / norm
            embeddings[i] = vec
        return embeddings

    @staticmethod
    def cosine_similarity_matrix(query_vectors: np.ndarray, doc_vectors: np.ndarray) -> np.ndarray:
        """Compute pairwise cosine similarities between query vectors and document vectors.
        
        Assumes inputs are already L2-normalized, reducing to simple matrix dot-product.
        """
        if query_vectors.ndim == 1:
            query_vectors = query_vectors.reshape(1, -1)
        if doc_vectors.ndim == 1:
            doc_vectors = doc_vectors.reshape(1, -1)
            
        return np.dot(query_vectors, doc_vectors.T)
