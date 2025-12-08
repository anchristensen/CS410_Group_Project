"""
Fusion and Reranking Module for StackOverflow Duplicate Question Retrieval

This module implements various fusion strategies to combine results from
BM25/RM3 (lexical) and Dense (semantic) retrievers.

Fusion Methods:
- Reciprocal Rank Fusion (RRF): Rank-based fusion
- Linear Combination: Weighted score-based fusion with normalization
- Hybrid: Configurable combination of both methods

Author: CS410 Group Project Team
"""

import json
from typing import List, Dict, Any, Optional, Callable
import numpy as np


class FusionReranker:
    """
    Combines and reranks results from multiple retrieval systems.

    Supports multiple fusion strategies including RRF and weighted score combination.
    """

    def __init__(
        self,
        fusion_method: str = "rrf",
        rrf_k: int = 60,
        bm25_weight: float = 0.5,
        dense_weight: float = 0.5,
        normalization: str = "min-max"
    ):
        """
        Initialize the FusionReranker.

        Args:
            fusion_method: Fusion strategy - "rrf", "linear", or "hybrid"
            rrf_k: RRF constant (typically 60)
            bm25_weight: Weight for BM25/RM3 scores (used in linear/hybrid)
            dense_weight: Weight for dense retrieval scores (used in linear/hybrid)
            normalization: Score normalization method - "min-max", "z-score", or "none"
        """
        self.fusion_method = fusion_method
        self.rrf_k = rrf_k
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        self.normalization = normalization

        # Validate weights sum to 1 for linear combination
        if fusion_method in ["linear", "hybrid"]:
            total_weight = bm25_weight + dense_weight
            if not np.isclose(total_weight, 1.0):
                print(f"Warning: Weights sum to {total_weight}, normalizing to 1.0")
                self.bm25_weight = bm25_weight / total_weight
                self.dense_weight = dense_weight / total_weight

    def fuse(
        self,
        bm25_results: List[Dict[str, Any]],
        dense_results: List[Dict[str, Any]],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Fuse results from BM25/RM3 and Dense retrievers.

        Args:
            bm25_results: Results from BM25/RM3 retriever
            dense_results: Results from dense retriever
            top_k: Number of top results to return

        Returns:
            List of fused and reranked results with unified format
        """
        if self.fusion_method == "rrf":
            return self._reciprocal_rank_fusion(bm25_results, dense_results, top_k)
        elif self.fusion_method == "linear":
            return self._linear_combination(bm25_results, dense_results, top_k)
        elif self.fusion_method == "hybrid":
            return self._hybrid_fusion(bm25_results, dense_results, top_k)
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")

    def _reciprocal_rank_fusion(
        self,
        bm25_results: List[Dict[str, Any]],
        dense_results: List[Dict[str, Any]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Reciprocal Rank Fusion (RRF) algorithm.

        RRF score = sum over all rankers of: 1 / (k + rank)
        where k is a constant (typically 60) and rank starts at 1.

        Reference: Cormack et al. "Reciprocal Rank Fusion outperforms Condorcet
        and individual Rank Learning Methods" (SIGIR 2009)
        """
        rrf_scores = {}
        metadata = {}

        # Process BM25/RM3 results
        for rank, result in enumerate(bm25_results, start=1):
            doc_id = self._extract_doc_id(result, source="bm25")
            rrf_score = 1.0 / (self.rrf_k + rank)

            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0.0
                metadata[doc_id] = {
                    "question_id": doc_id,
                    "title": self._extract_title(result, source="bm25"),
                    "body": self._extract_body(result, source="bm25"),
                    "bm25_rank": rank,
                    "bm25_score": result.get("score", 0.0),
                    "dense_rank": None,
                    "dense_score": None,
                    "raw": result.get("raw", result)
                }

            rrf_scores[doc_id] += rrf_score
            metadata[doc_id]["bm25_rank"] = rank
            metadata[doc_id]["bm25_score"] = result.get("score", 0.0)

        # Process Dense results
        for rank, result in enumerate(dense_results, start=1):
            doc_id = self._extract_doc_id(result, source="dense")
            rrf_score = 1.0 / (self.rrf_k + rank)

            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0.0
                metadata[doc_id] = {
                    "question_id": doc_id,
                    "title": self._extract_title(result, source="dense"),
                    "body": self._extract_body(result, source="dense"),
                    "bm25_rank": None,
                    "bm25_score": None,
                    "dense_rank": rank,
                    "dense_score": result.get("score", 0.0),
                    "raw": result.get("raw", result)
                }

            rrf_scores[doc_id] += rrf_score
            metadata[doc_id]["dense_rank"] = rank
            metadata[doc_id]["dense_score"] = result.get("score", 0.0)

        # Sort by RRF score and prepare output
        sorted_doc_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        fused_results = []
        for doc_id in sorted_doc_ids[:top_k]:
            result = metadata[doc_id].copy()
            result["fusion_score"] = rrf_scores[doc_id]
            result["fusion_method"] = "rrf"
            fused_results.append(result)

        return fused_results

    def _linear_combination(
        self,
        bm25_results: List[Dict[str, Any]],
        dense_results: List[Dict[str, Any]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Linear combination of normalized scores from both retrievers.

        Final score = bm25_weight * norm(bm25_score) + dense_weight * norm(dense_score)
        """
        # Normalize scores
        bm25_normalized = self._normalize_scores(bm25_results, "score")
        dense_normalized = self._normalize_scores(dense_results, "score")

        # Create score dictionaries
        bm25_score_map = {}
        dense_score_map = {}
        all_metadata = {}

        for i, result in enumerate(bm25_results):
            doc_id = self._extract_doc_id(result, source="bm25")
            bm25_score_map[doc_id] = bm25_normalized[i]
            all_metadata[doc_id] = {
                "question_id": doc_id,
                "title": self._extract_title(result, source="bm25"),
                "body": self._extract_body(result, source="bm25"),
                "bm25_rank": i + 1,
                "bm25_score": result.get("score", 0.0),
                "bm25_normalized": bm25_normalized[i],
                "dense_rank": None,
                "dense_score": None,
                "dense_normalized": 0.0,
                "raw": result.get("raw", result)
            }

        for i, result in enumerate(dense_results):
            doc_id = self._extract_doc_id(result, source="dense")
            dense_score_map[doc_id] = dense_normalized[i]

            if doc_id not in all_metadata:
                all_metadata[doc_id] = {
                    "question_id": doc_id,
                    "title": self._extract_title(result, source="dense"),
                    "body": self._extract_body(result, source="dense"),
                    "bm25_rank": None,
                    "bm25_score": None,
                    "bm25_normalized": 0.0,
                    "dense_rank": i + 1,
                    "dense_score": result.get("score", 0.0),
                    "dense_normalized": dense_normalized[i],
                    "raw": result.get("raw", result)
                }
            else:
                all_metadata[doc_id]["dense_rank"] = i + 1
                all_metadata[doc_id]["dense_score"] = result.get("score", 0.0)
                all_metadata[doc_id]["dense_normalized"] = dense_normalized[i]

        # Compute combined scores
        combined_scores = {}
        for doc_id in all_metadata.keys():
            bm25_norm = bm25_score_map.get(doc_id, 0.0)
            dense_norm = dense_score_map.get(doc_id, 0.0)
            combined_scores[doc_id] = (
                self.bm25_weight * bm25_norm + self.dense_weight * dense_norm
            )
            all_metadata[doc_id]["fusion_score"] = combined_scores[doc_id]

        # Sort and return top-k
        sorted_doc_ids = sorted(combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True)

        fused_results = []
        for doc_id in sorted_doc_ids[:top_k]:
            result = all_metadata[doc_id].copy()
            result["fusion_method"] = "linear"
            fused_results.append(result)

        return fused_results

    def _hybrid_fusion(
        self,
        bm25_results: List[Dict[str, Any]],
        dense_results: List[Dict[str, Any]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Hybrid fusion: combines RRF and linear score combination.

        Final score = 0.5 * RRF_score + 0.5 * Linear_score
        """
        # Get RRF results
        rrf_results = self._reciprocal_rank_fusion(bm25_results, dense_results, top_k=100)
        rrf_score_map = {r["question_id"]: r["fusion_score"] for r in rrf_results}

        # Get linear combination results
        linear_results = self._linear_combination(bm25_results, dense_results, top_k=100)
        linear_score_map = {r["question_id"]: r["fusion_score"] for r in linear_results}

        # Normalize both score sets
        rrf_scores_list = list(rrf_score_map.values())
        linear_scores_list = list(linear_score_map.values())

        if len(rrf_scores_list) > 0:
            rrf_normalized = self._normalize_scores_array(rrf_scores_list)
            rrf_score_map = {doc_id: rrf_normalized[i] for i, doc_id in enumerate(rrf_score_map.keys())}

        if len(linear_scores_list) > 0:
            linear_normalized = self._normalize_scores_array(linear_scores_list)
            linear_score_map = {doc_id: linear_normalized[i] for i, doc_id in enumerate(linear_score_map.keys())}

        # Combine
        all_doc_ids = set(rrf_score_map.keys()) | set(linear_score_map.keys())
        hybrid_scores = {}

        for doc_id in all_doc_ids:
            rrf_score = rrf_score_map.get(doc_id, 0.0)
            linear_score = linear_score_map.get(doc_id, 0.0)
            hybrid_scores[doc_id] = 0.5 * rrf_score + 0.5 * linear_score

        # Get metadata from linear results (more complete)
        metadata_map = {r["question_id"]: r for r in linear_results}

        # Sort and prepare output
        sorted_doc_ids = sorted(hybrid_scores.keys(), key=lambda x: hybrid_scores[x], reverse=True)

        fused_results = []
        for doc_id in sorted_doc_ids[:top_k]:
            if doc_id in metadata_map:
                result = metadata_map[doc_id].copy()
                result["fusion_score"] = hybrid_scores[doc_id]
                result["fusion_method"] = "hybrid"
                result["rrf_component"] = rrf_score_map.get(doc_id, 0.0)
                result["linear_component"] = linear_score_map.get(doc_id, 0.0)
                fused_results.append(result)

        return fused_results

    def _normalize_scores(
        self,
        results: List[Dict[str, Any]],
        score_key: str = "score"
    ) -> List[float]:
        """
        Normalize scores from a result list.

        Args:
            results: List of result dictionaries
            score_key: Key to extract scores from

        Returns:
            List of normalized scores
        """
        scores = [r.get(score_key, 0.0) for r in results]
        return self._normalize_scores_array(scores)

    def _normalize_scores_array(self, scores: List[float]) -> List[float]:
        """
        Normalize an array of scores based on the chosen normalization method.

        Args:
            scores: List of raw scores

        Returns:
            List of normalized scores
        """
        if len(scores) == 0:
            return []

        scores_array = np.array(scores)

        if self.normalization == "min-max":
            min_score = scores_array.min()
            max_score = scores_array.max()
            if max_score - min_score > 1e-10:
                normalized = (scores_array - min_score) / (max_score - min_score)
            else:
                normalized = np.ones_like(scores_array)

        elif self.normalization == "z-score":
            mean_score = scores_array.mean()
            std_score = scores_array.std()
            if std_score > 1e-10:
                normalized = (scores_array - mean_score) / std_score
                # Shift to positive range
                normalized = normalized - normalized.min()
                if normalized.max() > 1e-10:
                    normalized = normalized / normalized.max()
            else:
                normalized = np.ones_like(scores_array)

        elif self.normalization == "none":
            normalized = scores_array

        else:
            raise ValueError(f"Unknown normalization method: {self.normalization}")

        return normalized.tolist()

    def _extract_doc_id(self, result: Dict[str, Any], source: str) -> str:
        """
        Extract document ID from result, handling different formats.

        Args:
            result: Result dictionary
            source: Source system ("bm25" or "dense")

        Returns:
            Document ID as string
        """
        if source == "bm25":
            # BM25 results have "docid" field
            if "docid" in result:
                return str(result["docid"])
            # Fallback: try to parse from raw JSON
            if "raw" in result:
                try:
                    raw_data = json.loads(result["raw"]) if isinstance(result["raw"], str) else result["raw"]
                    return str(raw_data.get("question_id", raw_data.get("id", "unknown")))
                except:
                    pass

        elif source == "dense":
            # Dense results have "question_id" field
            if "question_id" in result:
                return str(result["question_id"])

        # Fallback: try common fields
        return str(result.get("id", result.get("question_id", result.get("docid", "unknown"))))

    def _extract_title(self, result: Dict[str, Any], source: str) -> str:
        """
        Extract question title from result.

        Args:
            result: Result dictionary
            source: Source system ("bm25" or "dense")

        Returns:
            Question title
        """
        if "title" in result:
            return result["title"]

        if "raw" in result:
            try:
                raw_data = json.loads(result["raw"]) if isinstance(result["raw"], str) else result["raw"]
                return raw_data.get("title", "No title")
            except:
                pass

        return "No title"

    def _extract_body(self, result: Dict[str, Any], source: str) -> str:
        """
        Extract question body from result.

        Args:
            result: Result dictionary
            source: Source system ("bm25" or "dense")

        Returns:
            Question body text
        """
        if "body" in result:
            return result["body"]

        if "raw" in result:
            try:
                raw_data = json.loads(result["raw"]) if isinstance(result["raw"], str) else result["raw"]
                return raw_data.get("body", "No body")
            except:
                pass

        return "No body"


def print_fused_results(results: List[Dict[str, Any]], max_results: int = 5):
    """
    Pretty print fused results with scores and metadata.

    Args:
        results: List of fused result dictionaries
        max_results: Maximum number of results to display
    """
    print(f"\n{'='*80}")
    print(f"FUSED RESULTS (Top {min(len(results), max_results)})")
    print(f"{'='*80}\n")

    for i, result in enumerate(results[:max_results], 1):
        print(f"[Rank {i}] Question ID: {result.get('question_id', 'N/A')}")
        print(f"Fusion Score: {result.get('fusion_score', 0.0):.4f} (method: {result.get('fusion_method', 'unknown')})")

        # Print component scores
        if result.get("bm25_score") is not None:
            print(f"  BM25 - Rank: {result.get('bm25_rank', 'N/A')}, Score: {result.get('bm25_score', 0.0):.4f}")
        if result.get("dense_score") is not None:
            print(f"  Dense - Rank: {result.get('dense_rank', 'N/A')}, Score: {result.get('dense_score', 0.0):.4f}")

        # Print question details
        title = result.get("title", "No title")
        if len(title) > 100:
            title = title[:97] + "..."
        print(f"Title: {title}")

        body = result.get("body", "No body")
        if len(body) > 150:
            body = body[:147] + "..."
        print(f"Body: {body}")
        print(f"{'-'*80}\n")


# Example usage and testing
if __name__ == "__main__":
    # Mock data for testing
    mock_bm25_results = [
        {"docid": "123", "score": 15.5, "raw": '{"question_id": 123, "title": "How to install pip?", "body": "I am having trouble..."}'},
        {"docid": "456", "score": 12.3, "raw": '{"question_id": 456, "title": "Pip installation error", "body": "Error when running pip..."}'},
        {"docid": "789", "score": 10.1, "raw": '{"question_id": 789, "title": "Python package manager", "body": "Which package manager..."}'},
    ]

    mock_dense_results = [
        {"question_id": 456, "score": 0.92, "title": "Pip installation error", "body": "Error when running pip...", "raw": {}},
        {"question_id": 999, "score": 0.88, "title": "Install packages python", "body": "How do I install packages...", "raw": {}},
        {"question_id": 123, "score": 0.85, "title": "How to install pip?", "body": "I am having trouble...", "raw": {}},
    ]

    print("Testing Fusion Methods\n")

    # Test RRF
    print("1. Reciprocal Rank Fusion (RRF)")
    print("-" * 80)
    reranker_rrf = FusionReranker(fusion_method="rrf", rrf_k=60)
    rrf_results = reranker_rrf.fuse(mock_bm25_results, mock_dense_results, top_k=5)
    print_fused_results(rrf_results)

    # Test Linear Combination
    print("\n2. Linear Combination (50-50 weight)")
    print("-" * 80)
    reranker_linear = FusionReranker(
        fusion_method="linear",
        bm25_weight=0.5,
        dense_weight=0.5,
        normalization="min-max"
    )
    linear_results = reranker_linear.fuse(mock_bm25_results, mock_dense_results, top_k=5)
    print_fused_results(linear_results)

    # Test Hybrid
    print("\n3. Hybrid Fusion (RRF + Linear)")
    print("-" * 80)
    reranker_hybrid = FusionReranker(
        fusion_method="hybrid",
        rrf_k=60,
        bm25_weight=0.5,
        dense_weight=0.5,
        normalization="min-max"
    )
    hybrid_results = reranker_hybrid.fuse(mock_bm25_results, mock_dense_results, top_k=5)
    print_fused_results(hybrid_results)
