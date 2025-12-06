# We should not be using the API for this.
# StackOverflow explicitly recommends not using their live API for bulk research.
# They provide a full offline dump, updated monthly:
# The Stack Overflow Data Dump (Archive.org)
# https://archive.org/details/stackexchange
# I tried to update with safe fetch limits to try and then run it anyways and got my ip blocked........ so I'm trying to redo this with a download of posts

import requests
from bs4 import BeautifulSoup
from stackapi import StackAPI
from datetime import datetime
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import html
import time

# #Initalizations for NLTK
first_time = False
if first_time:
    nltk.download('stopwords')
    nltk.download('punkt_tab')
stop_words = set(stopwords.words('english'))

# #initialization & paramters for stackoverflow API
site = StackAPI('stackoverflow')
site.page_size = 10
site.max_pages = 1                  # right now only fetching 10 items, adjust later once done debugging/etc. need >= thousands for our retrieval tool - note that we may run into rate limits
retrieval_start_date = datetime(2010,1,1)
retrieval_end_date = datetime(2024,12,31)

def as_tstamp(datetime_obj):
    return int(datetime_obj.timestamp())

# we can't just do the retrieve_content for a bulk pull of data - see info/potential bans if we pull tm at once here https://api.stackexchange.com/docs/throttle
# ----------------------------------
def safe_fetch(method, **kwargs):
    """
    Wrapper around site.fetch() that:
    - Handles API backoff responses
    - Prevents rapid-fire requests
    - Retries on temporary errors
    """
    while True:
        try:
            response = site.fetch(method, **kwargs)

            # If API tells us to back off:
            if "backoff" in response:
                backoff_time = response["backoff"]
                # Fix infinite-loop case ??
                if backoff_time == 0:
                    backoff_time = 2  # sleep at least 2 seconds
                print(f"[API BACKOFF] Sleeping for {backoff_time} seconds...")
                time.sleep(backoff_time)
                continue  # retry fetch after backoff

            # Normal rate control: sleep 0.2 sec
            time.sleep(0.2)

            return response

        except Exception as e:
            print(f"[API ERROR] {e}. Waiting 2 seconds before retry...")
            time.sleep(2)
            continue


def retrieve_content_bulk(start=datetime(2024,1,1), end=datetime(2024,1,2), query=None):
    print('Retrieving data...')
    start_ts = as_tstamp(start)
    end_ts = as_tstamp(end)
    
    all_items = []
    page = 1
    
    while True:
        print(f"\n--- Fetching page {page} ---")
        
        try:   
            if query:
                response = safe_fetch('search/advanced', 
                                q=query, 
                                fromdate=start_ts, 
                                todate=end_ts, 
                                filter='withbody',
                                page=page)
            else:   
                response = safe_fetch('questions', 
                                fromdate=start_ts, 
                                todate=end_ts, 
                                filter='withbody',
                                page=page)
                
            items = response.get("items", [])
            
            # stop if API returns no more results
            if not items:
                print("No more items. Stopping.")
                break

            all_items.extend(items)
            print(f"Retrieved {len(items)} items (Total: {len(all_items)})")

            if page >= site.max_pages:
                print("Reached max_pages. Stopping.")
                break

            page += 1            
            time.sleep(0.2)  # Small delay to avoid rate issues
            
        except Exception as e:
            print(f"[ERROR ON PAGE {page}] {e}")
            print("Waiting 3 seconds and retrying page...")
            time.sleep(3)
            # Retry SAME page again (do NOT page += 1)
            continue

    print(f"\nSuccessfully retrieved {len(all_items)} total records.")
    return all_items

def clean_html_thorough(text):
    soup = BeautifulSoup(text, "html.parser")
    for element in soup(["script", "style", "img", "iframe", "noscript"]):
            element.decompose()
    return soup.get_text(separator=" ")

def clean_html(text):
    text_html_removed = BeautifulSoup(text, "lxml").text
    text_html_removed = html.unescape(text_html_removed)             # originally said "text_html_removed = html.unescape(text)" but should be html.unescape(text_html_removed)
    text_html_removed = clean_html_thorough(text_html_removed)       # sim thing, chage back if I'm wrong though
    return text_html_removed

def tokenization(text):
    tokens = word_tokenize(text)
    tokens_cleaned = [word.lower() for word in tokens if word.lower() not in stop_words]
    return tokens_cleaned

def retrieve_answer_text(a_id):
    answer = safe_fetch('answers', ids=[a_id], filter='withbody')
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


content = retrieve_content_bulk(start=datetime(2024,1,1), end=datetime(2024,1,2), query="python installation error")
query = "python installation error"   # good to use to test small amount of similar results but note that we will want a large corpus of diverse posts with SOME duplicates built once with thousandas of results, not to use this in final result
content = retrieve_process_data(query)
print(content[0])