from flask import Flask, render_template, request, jsonify
from bm25_rm3_index import build_index, search_similar_questions
from dense_retriever import dense_search, build_dense_index
from fusion_reranker import FusionReranker
import json

app = Flask(__name__, template_folder='ui', static_folder='ui')

# helper to calculate overlap score
def calculate_overlap(query, text):
    # split into words
    query_words = query.lower().split()
    text_words = text.lower().split()

    # common stopwords; shouldn't count
    stopwords = ['the', 'a', 'an', 'and', 'or', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'is', 'it']

    filtered_query = []
    for word in query_words:
        if word not in stopwords:
            filtered_query.append(word)

    filtered_text = []
    for word in text_words:
        if word not in stopwords:
            filtered_text.append(word)

    # calculate overlap
    overlap_count = 0
    for word in filtered_query:
        if word in filtered_text:
            overlap_count += 1

    # calculate overlap score
    if len(filtered_query) == 0:
        return 0.0

    score = overlap_count / len(filtered_query)
    return round(score, 2)

# helper to format BM25 results
def format_bm25_results(bm25_results, query):
    results = []
    for result in bm25_results:
        raw_data = json.loads(result['raw'])
        contents = raw_data.get('contents', '')
        lines = contents.split('\n')
        title = ''
        question = ''
        answer = ''

        if len(lines) > 0:
            title = lines[0]
        if len(lines) > 1:
            question = lines[1]
        if len(lines) > 2:
            answer = '\n'.join(lines[2:])
        else:
            answer = 'N/A'

        combined_text = title + ' ' + question
        overlap_score = calculate_overlap(query, combined_text)
        results.append({
            'id': raw_data.get('id'),
            'title': title,
            'question': question,
            'answer': answer,
            'score': round(result['score'], 2),
            'overlap': overlap_score
        })
    return results

# helper to format dense results
def format_dense_results(dense_results, query):
    results = []
    for result in dense_results:
        title = result.get('title', '')
        body = result.get('body', '')
        lines = body.split('\n')

        question = ''
        answer = ''
        if len(lines) > 0:
            question = lines[0]
        if len(lines) > 1:
            answer = '\n'.join(lines[1:])
        else:
            answer = 'N/A'

        combined_text = title + ' ' + question
        overlap_score = calculate_overlap(query, combined_text)
        results.append({
            'id': result.get('question_id', ''),
            'title': title,
            'question': question,
            'answer': answer,
            'score': round(result['score'], 2),
            'overlap': overlap_score
        })
    return results

# helper to format fusion results
def format_fusion_results(fusion_results, query):
    results = []
    for result in fusion_results:
        title = result.get('title', '')
        body = result.get('body', '')
        lines = body.split('\n')

        question = ''
        answer = ''
        if len(lines) > 0:
            question = lines[0]
        if len(lines) > 1:
            answer = '\n'.join(lines[1:])
        else:
            answer = 'N/A'

        combined_text = title + ' ' + question
        overlap_score = calculate_overlap(query, combined_text)
        results.append({
            'id': result.get('question_id', ''),
            'title': title,
            'question': question,
            'answer': answer,
            'score': round(result.get('fusion_score', 0.0), 2),
            'overlap': overlap_score
        })
    return results

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form.get('query')
    k = int(request.form.get('k'))

    all_results = {}
    # 1. BM25/RM3
    print(f"\nBM25/RM3...")
    bm25_results = search_similar_questions(query, top_k=k, use_rm3=True)
    all_results['bm25'] = format_bm25_results(bm25_results, query)
    # 2. Dense
    print(f"\nDense Retrieval...")
    dense_results = dense_search(query, top_k=k)
    all_results['dense'] = format_dense_results(dense_results, query)
    # 3. RRF
    print(f"\nRRF Fusion...")
    rrf_reranker = FusionReranker(fusion_method='rrf', rrf_k=60)
    rrf_results = rrf_reranker.fuse(bm25_results, dense_results, top_k=k)
    all_results['rrf'] = format_fusion_results(rrf_results, query)
    # 4. Linear
    print(f"\nLinear Fusion...")
    linear_reranker = FusionReranker(
        fusion_method='linear',
        bm25_weight=0.5,
        dense_weight=0.5,
        normalization='min-max'
    )
    linear_results = linear_reranker.fuse(bm25_results, dense_results, top_k=k)
    all_results['linear'] = format_fusion_results(linear_results, query)
    # 5. Hybrid
    print(f"\nHybrid Fusion...")
    hybrid_reranker = FusionReranker(
        fusion_method='hybrid',
        rrf_k=60,
        bm25_weight=0.5,
        dense_weight=0.5,
        normalization='min-max'
    )
    hybrid_results = hybrid_reranker.fuse(bm25_results, dense_results, top_k=k)
    all_results['hybrid'] = format_fusion_results(hybrid_results, query)

    return jsonify({'all_results': all_results, 'query': query})

if __name__ == '__main__':
    print("Building BM25 index...")
    build_index()
    print("Building Dense index...")
    build_dense_index()
    print("Starting server...")
    app.run(debug=True)
