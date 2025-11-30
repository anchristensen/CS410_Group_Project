import requests
from bs4 import BeautifulSoup
from stackapi import StackAPI
from datetime import datetime
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import html

#Initalizations for NLTK
first_time = False
if first_time:
    nltk.download('stopwords')
    nltk.download('punkt_tab')
stop_words = set(stopwords.words('english'))

#initialization & paramters for stackoverflow API
site = StackAPI('stackoverflow')
site.page_size = 10
site.max_pages = 1
retrieval_start_date = datetime(2010,1,1)
retrieval_end_date = datetime(2024,12,31)

def as_tstamp(datetime_obj):
    return int(datetime_obj.timestamp())

def retrieve_content_bulk(start=datetime(2024,1,1), end=datetime(2024,1,2), query=None):
    print('Retrieving data...')
    start = as_tstamp(start)
    end = as_tstamp(end)
    if query:
        posts = site.fetch('search/advanced', 
                        q=query, 
                        fromdate=start, 
                        todate=end, 
                        filter='withbody')
    else:   
        posts = site.fetch('questions', 
                        fromdate=start, 
                        todate=end, 
                        filter='withbody')
    result = posts.get('items', [])
    num_records = len(result)
    print(f'Successfully retrieved {num_records} records')
    return result

def clean_html_thorough(text):
    soup = BeautifulSoup(text, "html.parser")
    for element in soup(["script", "style", "img", "iframe", "noscript"]):
            element.decompose()
    return soup.get_text(separator=" ")

def clean_html(text):
    text_html_removed = BeautifulSoup(text, "lxml").text
    text_html_removed = html.unescape(text)
    text_html_removed = clean_html_thorough(text)
    return text_html_removed

def tokenization(text):
    tokens = word_tokenize(text)
    tokens_cleaned = [word.lower() for word in tokens if word.lower() not in stop_words]
    return tokens_cleaned

def retrieve_answer_text(a_id):
    answer = site.fetch('answers', ids=[a_id], filter='withbody')
    answers = answer.get('items', [])
    answer_txt = answers[0]['body']
    return answer_txt

def format_data(entity):
    cleaned_body = clean_html(entity['body'])
    cleaned_title = clean_html(entity['title'])
    tokens = tokenization(cleaned_body)
    is_answered = 'accepted_answer_id' in entity
    structure = {
        'q_id': entity['question_id'],
        'title': cleaned_title,
        'body': cleaned_body,
        'tokens': tokens
    }
    if is_answered:
        a_id = entity['accepted_answer_id']
        a_text = clean_html(retrieve_answer_text(a_id))
        structure['a_id'] = a_id
        structure['answer'] =  a_text
    return structure

def structure_data(data_items):
    result = []
    for item in data_items:
        formatted_item = format_data(item)
        result.append(formatted_item)
    return result

def retrieve_process_data(query):
    #Retrieve the bulk data using the stackoverflow API:
    bulk_data = retrieve_content_bulk(retrieval_start_date, retrieval_end_date, query)

    # Structure the data as JSON lst; remove HTML elements 
    structured_data = structure_data(bulk_data)

    return structured_data


#content = retrieve_content_bulk(start=datetime(2024,1,1), end=datetime(2024,1,2), query="python installation error")
query = "python installation error"
content = retrieve_process_data(query)
print(content[0])