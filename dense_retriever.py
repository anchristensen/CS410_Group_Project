import os
import json
from typing import List, Dict, Any, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

DATA_DIR = "processed"
DENSE_INDEX = "dense_index.faiss"
DENSE_META = "dense_metadata.json"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_corpus(data_dir: str = DATA_DIR) -> Tuple[List[str], List[Dict[str, Any]]]:
    texts: List[str] = []
    meta: List[Dict[str, Any]] = []

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Processed data folder not found: {data_dir}")

    for root, _, files in os.walk(data_dir):
        for fname in files:
            if not fname.endswith(".jsonl"):
                continue
            path = os.path.join(root, fname)
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    title = obj.get("title") or ""
                    body = obj.get("body") or ""
                    text = title + " " + body
                    if not text.strip():
                        continue
                    texts.append(text)
                    meta.append(obj)

    if not texts:
        raise ValueError("No documents found in processed/ directory.")

    return texts, meta


def build_dense_index(
    data_dir: str = DATA_DIR,
    index_path: str = DENSE_INDEX,
    meta_path: str = DENSE_META,
    model_name: str = MODEL_NAME,
    batch_size: int = 64,
) -> None:
    if os.path.exists(index_path) and os.path.exists(meta_path):
        print("Dense index already exists. Skipping indexing.")
        return

    texts, metadata = load_corpus(data_dir)

    model = SentenceTransformer(model_name)
    all_embeddings: List[np.ndarray] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        emb = model.encode(batch, convert_to_numpy=True, show_progress_bar=True)
        all_embeddings.append(emb)

    X = np.vstack(all_embeddings).astype("float32")
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1
    X = X / norms

    dim = X.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(X)

    faiss.write_index(index, index_path)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f)


class DenseRetriever:
    def __init__(
        self,
        index_path: str = DENSE_INDEX,
        meta_path: str = DENSE_META,
        model_name: str = MODEL_NAME,
    ) -> None:
        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"Missing dense index at {index_path}. Run build_dense_index()."
            )
        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"Missing dense metadata at {meta_path}. Run build_dense_index()."
            )

        self.index = faiss.read_index(index_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)
        self.model = SentenceTransformer(model_name)

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        q = self.model.encode([query], convert_to_numpy=True).astype("float32")
        norms = np.linalg.norm(q, axis=1, keepdims=True)
        norms[norms == 0] = 1
        q = q / norms

        scores, idx = self.index.search(q, top_k)
        scores = scores[0]
        idx = idx[0]

        results: List[Dict[str, Any]] = []
        for score, i in zip(scores, idx):
            if i < 0 or i >= len(self.meta):
                continue
            m = self.meta[i]
            results.append(
                {
                    "score": float(score),
                    "question_id": m.get("question_id"),
                    "title": m.get("title"),
                    "body": m.get("body"),
                    "raw": m,
                }
            )

        return results


def dense_search(query: str, top_k: int = 10) -> List[Dict[str, Any]]:
    retriever = DenseRetriever()
    return retriever.search(query, top_k)


if __name__ == "__main__":
    if not os.path.exists(DENSE_INDEX) or not os.path.exists(DENSE_META):
        build_dense_index()
    r = DenseRetriever()
    q = "python import error"
    hits = r.search(q, 5)
    for h in hits:
        print(h["score"], h["title"])
