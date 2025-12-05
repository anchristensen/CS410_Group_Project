from bm25_rm3_index import build_index, search_similar_questions

def main():
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
