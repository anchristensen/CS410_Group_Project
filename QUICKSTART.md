# Quick Start Guide

## Installation Complete! ✅

Your virtual environment has been set up with all dependencies installed.

## How to Run

### Option 1: Using the Helper Script (Easiest)

```bash
./run.sh
```

### Option 2: Manual Commands

```bash
# Activate virtual environment
source venv/bin/activate

# Run the fusion reranking system
python main.py

# When done, deactivate
deactivate
```

### Option 3: Test Fusion Module Only

```bash
# Activate virtual environment
source venv/bin/activate

# Test fusion with mock data (no dataset download needed)
python fusion_reranker.py

# Deactivate
deactivate
```

## What Will Happen When You Run

When you run `./run.sh` or `python main.py`, the system will:

1. **Check Kaggle Config** - Verifies or creates Kaggle API credentials
2. **Download Data** - Downloads StackOverflow dataset (if not already present)
3. **Process Data** - Cleans HTML, tokenizes, merges Q&A
4. **Build BM25 Index** - Creates Lucene index for lexical search
5. **Build Dense Index** - Creates FAISS index for semantic search
6. **Run Test Query** - Searches for: "pip install error on python"
7. **Show BM25 Results** - Top 5 results from BM25/RM3
8. **Show Dense Results** - Top 5 results from Dense retrieval
9. **Apply Fusion** - Runs all 3 fusion methods:
   - Reciprocal Rank Fusion (RRF)
   - Linear Combination
   - Hybrid (RRF + Linear)

## Expected Output

You'll see detailed output comparing all three fusion methods, showing:
- Fusion scores
- Original BM25 and Dense ranks/scores
- Question titles and bodies
- Side-by-side comparison of ranking strategies

## Installed Dependencies

✅ Core Python packages:
- pandas, numpy, nltk, beautifulsoup4, requests

✅ IR/ML packages:
- pyserini (BM25/RM3)
- faiss-cpu (Dense retrieval)
- sentence-transformers (Embeddings)
- torch (PyTorch backend)
- scikit-learn (ML utilities)

✅ Data packages:
- kaggle (Dataset download)
- stackapi (Alternative API access)

## Troubleshooting

### If you get "Kaggle API not configured"

The system will try to generate it automatically. If it fails:
1. Go to kaggle.com → Your Account → API → Create New Token
2. Download `kaggle.json`
3. Place it in `~/.kaggle/kaggle.json`

### If you want to customize the query

Edit [main.py](main.py) line 35:
```python
test_query = "your custom query here"
```

### If you want to adjust fusion weights

Edit [main.py](main.py) lines 76-80:
```python
reranker = FusionReranker(
    fusion_method="linear",
    bm25_weight=0.7,  # Change this (0.0 to 1.0)
    dense_weight=0.3,  # Change this (sum should be 1.0)
    normalization="min-max"
)
```

## File Structure

```
CS410_Group_Project/
├── venv/                    # Virtual environment (created)
├── fusion_reranker.py       # Fusion module (NEW - implemented)
├── main.py                  # Main pipeline (UPDATED - with fusion)
├── bm25_rm3_index.py       # BM25/RM3 retrieval
├── dense_retriever.py      # Dense retrieval
├── data_pipeline_kaggle.py # Data processing
├── run.sh                  # Helper script (NEW)
├── QUICKSTART.md           # This file (NEW)
└── README.md               # Full documentation
```

## Next Steps

1. **Run the system**: `./run.sh`
2. **Review results**: See how fusion improves retrieval
3. **Experiment**: Try different queries and fusion parameters
4. **Evaluate**: Implement evaluation metrics (next phase)
5. **Build UI**: Create interactive interface (next phase)

## Need Help?

- Full documentation: See [README.md](README.md)
- Fusion details: See [fusion_reranker.py](fusion_reranker.py) docstrings
- Report issues: Contact team members

---

**Project**: StackOverflow Duplicate Question Retrieval System
**Course**: CS410 Text Information Systems, UIUC
**Team**: Taha Wasiq, Sparsh Singh, Jonathan Temkin, Annika Christensen, Sathvik Rajasekaran
