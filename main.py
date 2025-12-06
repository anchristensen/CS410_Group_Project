from bm25_rm3_index import build_index, search_similar_questions
import download_dataset
import generate_config
import data_pipeline_kaggle

def main():
    print("\n=== Checking Config File ===")
    generate_config.generate_kaggle_config()

    print("\n=== Checking data installation ===")
    download_dataset.install_data()

    print("\n=== Extracting and cleaning data")
    data = data_pipeline_kaggle.process_data()

    print("\n=== Building Lucene index (if needed) ===")
    build_index()

    test_query = "pip install error on python"
    print(f"\n=== Searching for: '{test_query}' ===")

    results = search_similar_questions(test_query, top_k=5, use_rm3=True)

    print("\n=== Top 5 Results ===")
    for r in results:
        print(f"DocID: {r['docid']}, Score: {r['score']}")
        print(f"Raw JSON: {r['raw']}")
        print("-" * 60)

if __name__ == "__main__":
    main()


