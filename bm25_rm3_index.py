import os
import json
import subprocess
from pyserini.search.lucene import LuceneSearcher
from data_pipeline_kaggle import process_data 


# so we can't do the retrieve_process_data with API limits and such, we need to download data
# but im running into issues with data size for instance from https://archive.org/details/stackexchange, the stackoverflow.com-Posts.7z 
# is 100 GB and we are not going to be able to upload that to github. need help with team ideas on fixing this
 

CORPUS_DIR = "processed_corpus"
INDEX_DIR = "indexes"
PROCESSED_DIR = "processed"  # For dense retriever

def dump_so_corpus(n_rows=10000):
    data = process_data(n_rows)

    if not os.path.exists(CORPUS_DIR):
        os.makedirs(CORPUS_DIR)

    for i, q in enumerate(data):
        doc = {
            "id": str(q["question_id"]),
            "contents": q["Title"] + "\n" + q["question_text"] + "\n" + q.get("answer_text", "")
        }

        with open(os.path.join(CORPUS_DIR, f"doc{i}.json"), "w") as f:
            json.dump(doc, f)

    # Also create JSONL format for dense retriever
    create_dense_corpus(data)


def create_dense_corpus(data=None, n_rows=10000):
    """Create JSONL corpus for dense retriever from processed data."""
    if data is None:
        data = process_data(n_rows)

    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)

    output_file = os.path.join(PROCESSED_DIR, "corpus.jsonl")

    with open(output_file, "w", encoding="utf-8") as f:
        for q in data:
            doc = {
                "question_id": q["question_id"],
                "title": q["Title"],
                "body": q["question_text"] + "\n" + q.get("answer_text", "")
            }
            f.write(json.dumps(doc) + "\n")

    print(f"Created JSONL corpus for dense retriever: {len(data)} documents")
            


def build_index():
    if os.path.exists(INDEX_DIR) and os.listdir(INDEX_DIR):
        print("Index already exists. Skipping indexing.")
        return

    # Create corpus if it doesn't exist
    if not os.path.exists(CORPUS_DIR) or not os.listdir(CORPUS_DIR):
        print("Creating corpus for indexing...")
        dump_so_corpus()

    cmd = [
        "python", "-m", "pyserini.index.lucene",
        "--collection", "JsonCollection",
        "--input", CORPUS_DIR,
        "--index", INDEX_DIR,
        "--generator", "DefaultLuceneDocumentGenerator",
        "--threads", "1",
        "--storePositions", "--storeDocvectors", "--storeRaw"
    ]

    print("Building BM25 index...")
    subprocess.run(cmd, check=True)
    print("BM25 index built successfully!")
    
    
def create_searcher(use_rm3=True):
    searcher = LuceneSearcher(INDEX_DIR)
    searcher.set_bm25(k1=1.2, b=0.75) # fine tune these parameters once large corpus not small testing

    if use_rm3:
        searcher.set_rm3(fb_terms=20, fb_docs=10, original_query_weight=0.5)

    return searcher


def search_similar_questions(query_text, top_k=10, use_rm3=True):
    searcher = create_searcher(use_rm3)

    hits = searcher.search(query_text, k=top_k)

    results = []
    for hit in hits:
        raw_json = searcher.doc(hit.docid).raw()

        results.append({
            "docid": hit.docid,
            "score": hit.score,
            "raw": raw_json
        })
    return results














