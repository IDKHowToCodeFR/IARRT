"""SentenceTransformers + FAISS retrieval over the idiom dictionary."""

import json
import os
from typing import Dict, List, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from utils.idiom_detection import normalize_text

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IDIOM_JSON = os.path.join(PROJECT_DIR, "data", "idiom_dictionary.json")


class IdiomRetriever:
    """Build a tiny FAISS index for German idioms and English meanings."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

        if os.path.exists(IDIOM_JSON):
            with open(IDIOM_JSON, "r") as f:
                self.idiom_dict = json.load(f)
        else:
            from utils.idiom_detection import IDIOM_DICT
            self.idiom_dict = IDIOM_DICT

        self.entries = [
            {"idiom": idiom, "meaning": meaning}
            for idiom, meaning in self.idiom_dict.items()
        ]
        texts = [f"{entry['idiom']} means {entry['meaning']}" for entry in self.entries]
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        self.embeddings = np.asarray(embeddings, dtype="float32")

        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.index.add(self.embeddings)

    def retrieve(self, query: str, k: int = 3) -> List[Dict[str, float]]:
        """Return top-k idioms and cosine-like similarity scores."""
        if query is None:
            return []
        query_text = normalize_text(query)
        if not query_text:
            return []
            
        print(f"DEBUG: Retrieving for '{query}' -> normalized: '{query_text}'")
        query_embedding = self.model.encode([query_text], normalize_embeddings=True)
        query_embedding = np.asarray(query_embedding, dtype="float32")

        scores, indices = self.index.search(query_embedding, min(k, len(self.entries)))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            entry = dict(self.entries[int(idx)])
            entry["score"] = float(score)
            results.append(entry)
        return results


_DEFAULT_RETRIEVER: Optional[IdiomRetriever] = None


def get_retriever() -> IdiomRetriever:
    """Create the retriever once and reuse it."""
    global _DEFAULT_RETRIEVER
    if _DEFAULT_RETRIEVER is None:
        _DEFAULT_RETRIEVER = IdiomRetriever()
    return _DEFAULT_RETRIEVER


def retrieve(query: str, k: int = 3) -> List[Dict[str, float]]:
    """Convenience function for scripts."""
    return get_retriever().retrieve(query, k=k)
