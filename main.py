from bm25_rm3_index import build_index, search_similar_questions
import download_dataset
import generate_config
import data_pipeline_kaggle

try:
    from dense_retriever import dense_search, build_dense_index
except:
    dense_search = None
    build_dense_index = None

try:
    from fusion_reranker import FusionReranker, print_fused_results
except:
    FusionReranker = None
    print_fused_results = None


def main():
    print("\n=== Checking Config File ===")
    generate_config.generate_kaggle_config()

    print("\n=== Checking data installation ===")
    download_dataset.install_data()

    print("\n=== Extracting and cleaning data")
    data = data_pipeline_kaggle.process_data(n_rows=10000)

    print("\n=== Building Lucene index (if needed) ===")
    build_index()

    # Try to build dense index
    dense_available = False
    if build_dense_index is not None:
        try:
            print("\n=== Building Dense index (if needed) ===")
            build_dense_index()
            dense_available = True
        except Exception as e:
            print(f"Warning: Could not build dense index: {e}")
            print("Dense retrieval will be skipped.")

    test_query = "pip install error on python"
    print(f"\n=== Searching for: '{test_query}' ===")

    # Retrieve results from both systems
    bm25_results = search_similar_questions(test_query, top_k=10, use_rm3=True)

    print("\n=== Top 5 Results (BM25/RM3) ===")
    for r in bm25_results[:5]:
        print(f"DocID: {r['docid']}, Score: {r['score']}")
        print(f"Raw JSON: {r['raw'][:100]}...")
        print("-" * 60)

    dense_results = None
    if dense_search is not None and dense_available:
        try:
            dense_results = dense_search(test_query, top_k=10)
            print("\n=== Top 5 Results (Dense) ===")
            for r in dense_results[:5]:
                title = r.get("title") or ""
                print(f"Score: {r['score']:.4f}, Title: {title[:80]}")
        except Exception as e:
            print(f"Warning: Dense search failed: {e}")
            dense_results = None

    # Fusion/Reranking
    if FusionReranker is not None and dense_results is not None:
        print("\n" + "="*80)
        print("APPLYING FUSION AND RERANKING")
        print("="*80)

        # Try different fusion methods
        fusion_methods = [
            ("rrf", "Reciprocal Rank Fusion (RRF)"),
            ("linear", "Linear Combination (BM25: 50%, Dense: 50%)"),
            ("hybrid", "Hybrid (RRF + Linear)")
        ]

        for method, description in fusion_methods:
            print(f"\n{'='*80}")
            print(f"Fusion Method: {description}")
            print(f"{'='*80}")

            if method == "rrf":
                reranker = FusionReranker(fusion_method="rrf", rrf_k=60)
            elif method == "linear":
                reranker = FusionReranker(
                    fusion_method="linear",
                    bm25_weight=0.5,
                    dense_weight=0.5,
                    normalization="min-max"
                )
            else:  # hybrid
                reranker = FusionReranker(
                    fusion_method="hybrid",
                    rrf_k=60,
                    bm25_weight=0.5,
                    dense_weight=0.5,
                    normalization="min-max"
                )

            fused_results = reranker.fuse(bm25_results, dense_results, top_k=5)
            print_fused_results(fused_results, max_results=5)
    else:
        if FusionReranker is None:
            print("\nWarning: FusionReranker not available. Install required dependencies.")
        if dense_results is None:
            print("\nWarning: Dense retrieval not available. Fusion requires both BM25 and Dense results.")


if __name__ == "__main__":
    main()
