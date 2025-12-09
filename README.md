# StackOverflow Duplicate Question Retrieval System

CS410 UIUC Group Project - Information Retrieval System for Detecting Duplicate Questions on Stack Overflow

## Team Members

- **Taha Wasiq** (twasiq2)
- **Sparsh Singh** (ss85)
- **Jonathan Temkin** (jtemkin3)
- **Annika Christensen** (annikac7) - Project Coordinator
- **Sathvik Rajasekaran** (sathvik4)

## Quick Start

```bash
# 1. Create virtual environment and install dependencies
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install pandas numpy nltk beautifulsoup4 lxml
pip install pyserini faiss-cpu sentence-transformers
pip install kaggle stackapi requests flask

# 2. Set up Kaggle credentials (place kaggle.json in ~/.kaggle/)
# Or run: python generate_config.py

# 3. Run the UI
python app.py

# Or use the helper script (activates venv automatically):
./run.sh
```
This will download data, build indexes, and run a test query with all three fusion methods (RRF, Linear, Hybrid). Results are displayed in the terminal.

## Project Overview

This project develops an intelligent duplicate question retrieval system for Stack Overflow. The system helps users find existing answers to their questions and reduces redundant posts by combining multiple retrieval approaches.

### Problem Statement

Stack Overflow often contains multiple versions of the same question, which:
- Scatters useful answers across duplicate threads
- Increases maintenance overhead for moderators
- Creates inefficient search experiences for developers

### Our Solution

We built a multi-strategy retrieval system that combines:
- **Lexical retrieval** (BM25 with RM3 query expansion)
- **Semantic retrieval** (Dense embeddings with Sentence-BERT)
- **Fusion/Reranking** (Reciprocal Rank Fusion and weighted combination)

This hybrid approach provides better duplicate detection than any single method alone.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Data Sources                              │
│  • Kaggle StackOverflow Dataset (Questions, Answers, Tags)  │
│  • StackOverflow API (optional)                             │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              Data Processing Pipeline                        │
│  • HTML cleaning & text preprocessing                        │
│  • Question-Answer merging                                   │
│  • Tokenization & stopword removal                          │
│  • Deduplication                                            │
└────────────────────┬────────────────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
┌──────────────────┐    ┌──────────────────┐
│  BM25/RM3 Index  │    │  Dense Index     │
│  (Pyserini/      │    │  (FAISS +        │
│   Lucene)        │    │   Sentence-BERT) │
└────────┬─────────┘    └────────┬─────────┘
         │                       │
         ▼                       ▼
┌──────────────────┐    ┌──────────────────┐
│ Lexical Search   │    │ Semantic Search  │
│ • BM25 ranking   │    │ • Cosine sim     │
│ • RM3 expansion  │    │ • Vector search  │
└────────┬─────────┘    └────────┬─────────┘
         │                       │
         └───────────┬───────────┘
                     ▼
         ┌────────────────────────┐
         │   Fusion & Reranking   │
         │  • RRF (Rank-based)    │
         │  • Linear (Score-based)│
         │  • Hybrid (Combined)   │
         └────────┬───────────────┘
                  ▼
         ┌────────────────────────┐
         │   Final Ranked Results │
         │  with Similarity Scores│
         └────────────────────────┘
```

## Implemented Components

### 1. Data Pipeline (Kaggle)
**File**: `data_pipeline_kaggle.py`

**Features**:
- Downloads StackOverflow dataset from Kaggle
- Merges Questions, Answers, and Tags
- Cleans HTML tags and markup
- Tokenizes text using NLTK
- Removes stopwords and duplicates
- Outputs processed JSON/JSONL for indexing

**Status**: ✅ Complete

### 2. Data Pipeline (API) - Alternative
**File**: `data_pipeline_stack_api.py`

**Features**:
- Direct data retrieval from StackOverflow API
- Rate limiting and backoff handling
- Custom query support
- Date range filtering

**Status**: ✅ Complete (not actively used due to API limits)

### 3. BM25/RM3 Retrieval
**File**: `bm25_rm3_index.py`

**Features**:
- Lucene-based indexing via Pyserini
- BM25 ranking with configurable parameters (k1=1.2, b=0.75)
- RM3 pseudo-relevance feedback for query expansion
- Returns ranked list with BM25 scores

**Algorithm**: Okapi BM25 + RM3
- BM25: Probabilistic term weighting based on term frequency and document length
- RM3: Expands query with terms from top-ranked documents

**Status**: ✅ Complete

### 4. Dense Retrieval
**File**: `dense_retriever.py`

**Features**:
- FAISS vector index for efficient similarity search
- Sentence-BERT embeddings (all-MiniLM-L6-v2)
- 384-dimensional dense vectors
- Cosine similarity ranking
- Batch encoding for performance

**Algorithm**: Neural embedding-based retrieval
- Encodes questions into dense vectors
- Uses inner product search (normalized for cosine similarity)

**Status**: ✅ Complete

### 5. Fusion & Reranking
**File**: `fusion_reranker.py`

**Features**:
- **Reciprocal Rank Fusion (RRF)**: Combines rankings from multiple systems
- **Linear Combination**: Weighted score fusion with normalization
- **Hybrid Method**: Combines RRF and linear approaches
- Flexible score normalization (min-max, z-score)
- Rich metadata tracking (ranks and scores from both systems)

**Algorithms**:

**RRF Formula**:
```
RRF_score(d) = Σ (1 / (k + rank_i(d)))
```
where k=60 (constant), rank_i is document rank in system i

**Linear Combination**:
```
score(d) = w1 × norm(score_bm25(d)) + w2 × norm(score_dense(d))
```

**Status**: ✅ Complete

### 6. Dataset Management
**Files**: `download_dataset.py`, `generate_config.py`

**Features**:
- Automatic Kaggle dataset download
- Kaggle API authentication
- Config file generation
- Dataset verification

**Status**: ✅ Complete

### 7. Main Orchestrator
**File**: `main.py`

**Features**:
- End-to-end pipeline execution
- Automatic index building
- Multi-method comparison
- Example query demonstration

**Status**: ✅ Complete

## Installation & Setup

### Prerequisites

```bash
# Python 3.8 or higher
python --version

# Required packages
pip install -r requirements.txt
```

### Required Dependencies

```bash
pip install pandas numpy nltk beautifulsoup4
pip install pyserini faiss-cpu sentence-transformers
pip install kaggle stackapi requests
```

### Kaggle API Setup

1. Create a Kaggle account and get API credentials
2. Place `kaggle.json` in `~/.kaggle/` directory, OR
3. The code will automatically generate config from environment variables

## Usage

### Running the Full Pipeline

```bash
python main.py
```

This will:
1. ✅ Check and generate Kaggle config
2. ✅ Download StackOverflow dataset (if not present)
3. ✅ Process and clean the data
4. ✅ Build BM25/RM3 index
5. ✅ Build Dense embedding index
6. ✅ Run test query: "pip install error on python"
7. ✅ Display results from BM25/RM3
8. ✅ Display results from Dense retrieval
9. ✅ Apply and compare all 3 fusion methods

### Using Individual Components

#### BM25/RM3 Search Only

```python
from bm25_rm3_index import search_similar_questions

query = "How to fix pip installation error?"
results = search_similar_questions(query, top_k=10, use_rm3=True)

for r in results:
    print(f"DocID: {r['docid']}, Score: {r['score']}")
```

#### Dense Search Only

```python
from dense_retriever import dense_search

query = "How to fix pip installation error?"
results = dense_search(query, top_k=10)

for r in results:
    print(f"Score: {r['score']:.4f}, Title: {r['title']}")
```

#### Fusion & Reranking

```python
from bm25_rm3_index import search_similar_questions
from dense_retriever import dense_search
from fusion_reranker import FusionReranker, print_fused_results

query = "How to fix pip installation error?"

# Get results from both systems
bm25_results = search_similar_questions(query, top_k=10, use_rm3=True)
dense_results = dense_search(query, top_k=10)

# Method 1: Reciprocal Rank Fusion (RRF)
reranker_rrf = FusionReranker(fusion_method="rrf", rrf_k=60)
fused = reranker_rrf.fuse(bm25_results, dense_results, top_k=5)
print_fused_results(fused)

# Method 2: Linear Combination (custom weights)
reranker_linear = FusionReranker(
    fusion_method="linear",
    bm25_weight=0.6,
    dense_weight=0.4,
    normalization="min-max"
)
fused = reranker_linear.fuse(bm25_results, dense_results, top_k=5)
print_fused_results(fused)

# Method 3: Hybrid (RRF + Linear)
reranker_hybrid = FusionReranker(
    fusion_method="hybrid",
    rrf_k=60,
    bm25_weight=0.5,
    dense_weight=0.5,
    normalization="min-max"
)
fused = reranker_hybrid.fuse(bm25_results, dense_results, top_k=5)
print_fused_results(fused)
```

## Project Structure

```
CS410_Group_Project/
│
├── main.py                      # Main entry point
├── bm25_rm3_index.py           # BM25/RM3 lexical retrieval
├── dense_retriever.py          # Dense semantic retrieval
├── fusion_reranker.py          # Fusion and reranking methods
├── data_pipeline_kaggle.py     # Kaggle data processing
├── data_pipeline_stack_api.py  # StackOverflow API pipeline
├── download_dataset.py         # Dataset download automation
├── generate_config.py          # Kaggle config generation
│
├── data/                       # Raw CSV files
│   ├── Questions.csv
│   ├── Answers.csv
│   └── Tags.csv
│
├── processed/                  # Processed JSON/JSONL files
│   └── corpus.jsonl
│
├── indexes/                    # BM25/Lucene index files
│
├── dense_index.faiss          # FAISS vector index
├── dense_metadata.json        # Dense index metadata
│
└── README.md                   # This file
```

## Fusion Methods Explained

### 1. Reciprocal Rank Fusion (RRF)

**Best for**: Robust combination without score calibration

**How it works**: Assigns a score based on the rank of each document across systems
```
RRF_score = Σ (1 / (k + rank))
```

**Advantages**:
- No need to normalize scores
- Works well when score distributions differ significantly
- Simple and effective

**When to use**: Default choice for most scenarios

### 2. Linear Combination

**Best for**: When you want to control the importance of each system

**How it works**: Weighted sum of normalized scores
```
score = w1 × norm(bm25) + w2 × norm(dense)
```

**Advantages**:
- Fine-grained control over system weights
- Can favor lexical or semantic matching
- Flexible normalization options

**When to use**: When you know one system performs better for your use case

### 3. Hybrid Fusion

**Best for**: Maximum robustness and performance

**How it works**: Combines normalized RRF and linear scores
```
score = 0.5 × norm(RRF) + 0.5 × norm(Linear)
```

**Advantages**:
- Best of both worlds
- Most robust to different query types
- Handles edge cases well

**When to use**: When you want the best overall performance

## Evaluation Plan

### Metrics

We plan to evaluate using the following IR metrics:

1. **Precision@K**: Proportion of top-K results that are true duplicates
2. **Recall@K**: Proportion of true duplicates found in top-K
3. **Mean Reciprocal Rank (MRR)**: Average of 1/rank for first relevant result
4. **NDCG@K**: Normalized Discounted Cumulative Gain

### Ground Truth

- Use existing duplicate-question labels from Stack Overflow
- For each query with a known duplicate, check if our system ranks it in top-K
- Compare performance across all three fusion methods

### Evaluation Strategy

```python
# Pseudocode for evaluation
for query, true_duplicate_id in test_set:
    results = fusion_search(query, top_k=10)

    # Check if true duplicate appears in results
    ranks = [i for i, r in enumerate(results) if r['id'] == true_duplicate_id]

    if ranks:
        mrr += 1.0 / (ranks[0] + 1)
        precision_at_k += 1 if ranks[0] < k else 0

    # ... calculate other metrics
```

## Technical Details

### BM25 Parameters
- **k1**: 1.2 (term saturation parameter)
- **b**: 0.75 (length normalization)

### RM3 Parameters
- **fb_terms**: 20 (feedback terms for query expansion)
- **fb_docs**: 10 (feedback documents)
- **original_query_weight**: 0.5

### Dense Retrieval Parameters
- **Model**: sentence-transformers/all-MiniLM-L6-v2
- **Embedding dimension**: 384
- **Similarity metric**: Cosine (via normalized inner product)
- **Batch size**: 64

### Fusion Parameters
- **RRF k**: 60 (standard constant)
- **Default weights**: 50% BM25, 50% Dense
- **Normalization**: Min-max (default)

## Performance Considerations

1. **Index Building**: One-time operation, can take several minutes
2. **BM25 Search**: Fast (< 100ms per query)
3. **Dense Search**: Fast after index is built (< 50ms per query)
4. **Fusion**: Negligible overhead (< 10ms)

## Future Enhancements

- [ ] Implement evaluation metrics (Precision, Recall, MRR, NDCG)
- [ ] Add learned-to-rank (LTR) with machine learning
- [ ] Implement cross-encoder reranking for top results
- [ ] Add query-adaptive fusion weights
- [ ] Build web UI for interactive search
- [ ] Support for incremental index updates
- [ ] Multi-language support
- [ ] Integration with StackExchange network (not just StackOverflow)

## Troubleshooting

### Common Issues

**1. Kaggle API authentication fails**
```bash
# Solution: Generate kaggle.json with your credentials
python generate_config.py
```

**2. Dataset download fails**
```bash
# Solution: Check internet connection and Kaggle API limits
# Manual download: https://www.kaggle.com/datasets/stackoverflow/stacksample
```

**3. Dense retriever not working**
```bash
# Solution: Install FAISS and sentence-transformers
pip install faiss-cpu sentence-transformers
```

**4. Out of memory when building dense index**
```bash
# Solution: Reduce batch size in dense_retriever.py
# Or use smaller dataset subset
```

## References

1. **BM25**: Robertson, S., & Zaragoza, H. (2009). The Probabilistic Relevance Framework: BM25 and Beyond.
2. **RM3**: Abdul-Jaleel, N., et al. (2004). UMass at TREC 2004: Novelty and HARD.
3. **Reciprocal Rank Fusion**: Cormack, G., et al. (2009). Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods.
4. **Sentence-BERT**: Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.

## License

This project is for educational purposes as part of CS410 UIUC coursework.

## Acknowledgments

- Course: CS410 Text Information Systems, UIUC
- Dataset: [StackOverflow on Kaggle](https://www.kaggle.com/datasets/stackoverflow/stacksample)
- Libraries: Pyserini, FAISS, Sentence-Transformers, NLTK
