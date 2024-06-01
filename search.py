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
    try:
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
                {"role": "user", "content": f"Given this text, generate a comprehensive summary in about 1 sentence. If no text is provided, generate a comprehensive summary in about 1 sentence from the title. \n\nTEXT={text}\nTITLE={title}"}
            ],
            max_tokens=50
        )

        summary = completion.choices[0].message.content

        return title, summary
    except:
        return url, "No summary could be generated."


@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


@app.route('/search')
def search():
    stemmer = SnowballStemmer("english")
    data = shelve.open("inverted_index_total.shelve")

    # parse query
    query = request.args.get('query')
    start_time = time.time_ns()
    query_lst = query.split()
    for i in range(len(query_lst)):
        query_lst[i] = stemmer.stem(query_lst[i].lower())

    # parse docs
    docs = dict()
    for q in query_lst:
        docs[q] = data[q].copy()

    results = cosine_sim(query_lst, docs)
    end_time = time.time_ns()
    duration = (end_time - start_time) / 1000000
    data.close()

    summaries = []
    for result in results:
        title, summary = summarize(result[0])
        summaries.append({"title": title, "summary": summary, "url": result[0]})

    return jsonify(results=summaries, duration=duration)


def positions_check(query_items, match_list, max_distance=5):
    """
    Find all positions for each query item.
    Ensure item n positions are before item n+1.
    Returns set of valid docs that are close in proximity.
    """
    doc_positions = defaultdict(list)

    # collect positions
    for term in query_items:
        if term in match_list:
            postings = match_list[term]["postings"]
            for p in postings:
                doc_positions[p["docID"]].append(p["positions"])

    # check positions
    valid_docs = set()
    for docid, positions in doc_positions.items():
        found = False
        for i in range(len(positions) - 1):
            if found: break
            for j in range(i+1, len(positions)):
                if found: break
                for pos1 in positions[i]:
                    if found: break
                    for pos2 in positions[j]:
                        if pos1 < pos2 and abs(pos1-pos2) <= max_distance:
                            valid_docs.add(docid)
                            found = True
                            break
    
    with open("valid_docs.txt", 'w') as file:
        for doc in valid_docs:
            file.write(doc + "\n")

    return valid_docs

def hits(query_items, match_list):
    return None
    # read the urls within the match_list, find url that connect to these 

def cosine_sim(query_items, match_list):
    scores = defaultdict(float)
    doc_length = defaultdict(float)

    valid_docs = positions_check(query_items, match_list)

    for term in query_items:
        if term in match_list:
            key_word = match_list[term]
            idf = key_word["idf"]
            postings = key_word["postings"]
            for p in postings:
                docid = p["docID"]
                if docid in valid_docs:
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
    app.run(debug=True)