import shelve
from nltk.stem.snowball import SnowballStemmer
from collections import defaultdict
import math
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
import time
import os
from dotenv import load_dotenv

app = Flask(__name__)
CORS(app)

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)

def summarize(url):
    # get page content
    response = requests.get(url, verify=False)
    response.raise_for_status()
    
    # parse page for title and text
    soup = BeautifulSoup(response.text, 'html.parser')
    title = soup.title.string if soup.title else url
    paragraphs = soup.find_all('p')
    text = ' '.join([paragraphs[i].get_text() for i in range(min(2, len(paragraphs)))])

    # summarize w/ gpt
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Given this text, generate a comprehensive summary in about 1 sentence. If no text is provided, create a shorter summary from the title.'\n\nTEXT={text}\nTITLE={title}"}
        ],
        max_tokens=50
    )

    summary = completion.choices[0].message.content

    return title, summary


@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


@app.route('/search')
def search():
    stemmer = SnowballStemmer("english")
    query = request.args.get('query')
    # query = "elimin chen"
    query_lst = query.split()
    for i in range(len(query_lst)):
        query_lst[i] = stemmer.stem(query_lst[i].lower())
    doc = {"elimin": {'df': 2, 'idf': 1.1903316981702914, 
    'postings': [{'docID': 'https://www.informatics.uci.edu/informatics-department-receives-1-1m-to-study-socially-responsible-ai/', 'tf': 1, 'tf-idf': 1.1903316981702914}, 
                 {'docID': 'https://www.informatics.uci.edu/strava-map-exposes-weaknesses-in-understanding-complexities-of-pervasive-data/#content', 'tf': 1, 'tf-idf': 1.1903316981702914}]},
    "chen": {'df': 5, 'idf': 0.7923916894982539, 
             'postings': [{'docID': 'https://www.informatics.uci.edu/grad/student-profiles/mustafa-hussain/', 'tf': 1, 'tf-idf': 0.79}, 
                          {'docID': 'https://www.informatics.uci.edu/2019/04/#content', 'tf': 1, 'tf-idf': 0.7923916894}, 
                          {'docID': 'https://www.informatics.uci.edu/dourish-publishes-the-stuff-of-bits/', 'tf': 1, 'tf-idf': 0.7923916894982539}, 
                          {'docID': 'https://www.informatics.uci.edu/dourish-publishes-the-stuff-of-bits/#content', 'tf': 1, 'tf-idf': 0.7923916894982539}, 
                          {'docID': 'https://www.informatics.uci.edu/explore/faculty-profiles/#content', 'tf': 1, 'tf-idf': 0.79239168949825}]}
        }
    
    start_time = time.time()
    results = cosine_sim(query_lst, doc)
    end_time = time.time()
    duration = end_time - start_time

    summaries = []
    for result in results:
        title, summary = summarize(result[0])
        summaries.append({"title": title, "summary": summary, "url": result[0]})

    return jsonify(results=summaries, duration=duration)

def cosine_sim(query_items, match_list):
    scores = defaultdict(float)
    doc_length = defaultdict(float)
    for term in query_items:
        if term in match_list:
            key_word = match_list[term]
            idf = key_word["idf"]
            postings = key_word["postings"]
            for p in postings:
                docid = p["docID"]
                tf_idf = p["tf-idf"]
                if tf_idf > 0:
                    w_td = 1 + math.log10(tf_idf)
                else:
                    w_td = 0
                w_tq = idf
                scores[docid] += w_td * w_tq
                doc_length[docid] += (tf_idf ** 2) ** 0.5
    for d in scores:
        if doc_length[d] > 0:
            scores[d] /= doc_length[d]
    best_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:5]
    return best_scores


if __name__ == "__main__":
    # stemmer = SnowballStemmer("english")
    # data = shelve.open("inverted_index_total.shelve")
    # query = "elimin chen"
    # start_time = time.process_time_ns()
    # query_lst = query.split()
    # for i in range(len(query_lst)):
    #     query_lst[i] = stemmer.stem(query_lst[i].lower())
    # dict1 = dict()
    # for q in query_lst:
    #     dict1[q] = data[q].copy()
    # print(cosine_sim(query_lst, dict1))
    # end_time = time.process_time_ns()
    # print(f"Indexing time: {end_time - start_time}")
    data = shelve.open("inverted_index_test.shelve")
    data.close()
    app.run(debug=True)