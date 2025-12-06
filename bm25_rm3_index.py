import os
import json
import subprocess
from pyserini.search.lucene import LuceneSearcher
from extract_clean_data import retrieve_process_data 


# so we can't do the retrieve_process_data with API limits and such, we need to download data
# but im running into issues with data size for instance from https://archive.org/details/stackexchange, the stackoverflow.com-Posts.7z 
# is 100 GB and we are not going to be able to upload that to github. need help with team ideas on fixing this
 

CORPUS_DIR = "processed_corpus"
INDEX_DIR = "indexes"

def dump_so_corpus(query=None):
    data = retrieve_process_data(query)

    for i, q in enumerate(data):
        doc = {
            "id": str(q["q_id"]),
            "contents": q["title"] + "\n" + q["body"]
        }

        with open(os.path.join(CORPUS_DIR, f"doc{i}.json"), "w") as f:
            json.dump(doc, f)
            


def build_index():
    if os.path.exists(INDEX_DIR) and os.listdir(INDEX_DIR):
        print("Index already exists. Skipping indexing.")
        return

    cmd = [
        "python", "-m", "pyserini.index.lucene",
        "--collection", "JsonCollection",
        "--input", CORPUS_DIR,
        "--index", INDEX_DIR,
        "--generator", "DefaultLuceneDocumentGenerator",
        "--threads", "1",
        "--storePositions", "--storeDocvectors", "--storeRaw"
    ]

    subprocess.run(cmd, check=True)
    
    
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














