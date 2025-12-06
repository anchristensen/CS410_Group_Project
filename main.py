from bm25_rm3_index import build_index, search_similar_questions
import download_dataset
import generate_config
import data_pipeline_kaggle

try:
    from dense_retriever import dense_search, build_dense_index
except:
    dense_search = None
    build_dense_index = None


def main():
    print("\n=== Checking Config File ===")
    generate_config.generate_kaggle_config()

    print("\n=== Checking data installation ===")
    download_dataset.install_data()

    print("\n=== Extracting and cleaning data")
    data = data_pipeline_kaggle.process_data()

    print("\n=== Building Lucene index (if needed) ===")
    build_index()

    if build_dense_index is not None:
        if not hasattr(build_dense_index, "__file__"):
            pass
        else:
            try:
                build_dense_index()
            except:
                pass

    test_query = "pip install error on python"
    print(f"\n=== Searching for: '{test_query}' ===")

    results = search_similar_questions(test_query, top_k=5, use_rm3=True)

    print("\n=== Top 5 Results (BM25/RM3) ===")
    for r in results:
        print(f"DocID: {r['docid']}, Score: {r['score']}")
        print(f"Raw JSON: {r['raw']}")
        print("-" * 60)

    if dense_search is not None:
        dense_results = dense_search(test_query, top_k=5)
        print("\n=== Top 5 Results (Dense) ===")
        for r in dense_results:
            title = r.get("title") or ""
            print(r["score"], title)


if __name__ == "__main__":
    main()
