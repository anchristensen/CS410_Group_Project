import pandas as pd
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
first_time = False
if first_time:
    nltk.download('stopwords')
    nltk.download('punkt_tab')
stop_words = set(stopwords.words('english'))
import html
import warnings
warnings.filterwarnings("ignore")

#QUESTIONS
def get_question_data():
    pth_questions = r'data\Questions.csv'
    df_questions = pd.read_csv(pth_questions, encoding='latin1')
    return df_questions

#ANSWERS
def get_answer_data():
    pth_answers = r'data\Answers.csv'
    df_answers = pd.read_csv(pth_answers, encoding='latin1')
    return df_answers

#QUESTIONS + ANSWERS
def merge_data(df1, df2, id1, id2, merge_type):
    #merged_df = pd.merge(df_questions,df_answers, left_on='Id',  right_on='ParentId', how='inner')
    merged_df = pd.merge(df1,df2, left_on=id1,  right_on=id2, how=merge_type)
    return merged_df

def remove_dupes(df, id):
    #df = df.drop_duplicates(subset=['question_id'], keep='first')
    df = df.drop_duplicates(subset=[id], keep='first')
    return df

#CLEAN DATA
def clean_html_thorough(text):
    soup = BeautifulSoup(text, "html.parser")
    for element in soup(["script", "style", "img", "iframe", "noscript"]):
            element.decompose()
    return soup.get_text(separator=" ")

def clean_html(text):
    text_html_removed = BeautifulSoup(text, "lxml").text
    text_html_removed = html.unescape(text_html_removed)             
    text_html_removed = clean_html_thorough(text_html_removed)       
    return text_html_removed

def tokenization(text):
    tokens = word_tokenize(text)
    tokens_cleaned = [word.lower() for word in tokens if word.lower() not in stop_words]
    return tokens_cleaned

def clean_data(merged_df):
    merged_df['question_text'] = merged_df['question_text'].apply(clean_html)
    merged_df['answer_text'] = merged_df['answer_text'].apply(clean_html)
    merged_df['question_tokens'] = merged_df['question_text'].apply(tokenization)
    merged_df['answer_tokens'] = merged_df['answer_text'].apply(tokenization)
    return merged_df

col_names = {
    'Id_x': 'question_id',
    'Body_x': 'question_text', 
    'Id_y': 'answer_id', 
    'Body_y': 'answer_text', 
}

def process_data(n_rows):
    q_data = get_question_data()
    a_data = get_answer_data()
    #merged_df = pd.merge(df_questions,df_answers, left_on='Id',  right_on='ParentId', how='inner')
    q_and_a_data = merge_data(q_data, a_data, 'Id','ParentId', 'inner')
    q_and_a_data = q_and_a_data.head(n_rows)
    q_and_a_data = q_and_a_data.rename(columns=col_names, inplace=False)
    q_and_a_data = q_and_a_data[['question_id', 'Title', 'question_text', 'answer_text']]
    q_and_a_data = remove_dupes(q_and_a_data, 'question_id')
    q_and_a_data_cleaned = clean_data(q_and_a_data)
    data = q_and_a_data_cleaned.to_dict('records')
    return data

data = process_data(10000)
print(data[0:5])
