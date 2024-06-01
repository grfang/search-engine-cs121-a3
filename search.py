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
import warnings
import urllib3
from urllib3.exceptions import InsecureRequestWarning

# Suppress only the single InsecureRequestWarning from urllib3
warnings.simplefilter('ignore', InsecureRequestWarning)

app = Flask(__name__)
CORS(app)

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)

with shelve.open("a_c_index.shelve") as a_c_dict:
    a_c = dict(a_c_dict)
print("Opened a_c")
with shelve.open("d_f_index.shelve") as d_f_dict:
    d_f = dict(d_f_dict)
print("Opened d_f")
with shelve.open("g_i_index.shelve") as g_i_dict:
    g_i = dict(g_i_dict)
print("Opened g_i")
with shelve.open("j_l_index.shelve") as j_l_dict:
    j_l = dict(j_l_dict)
print("Opened j_l")
with shelve.open("m_o_index.shelve") as m_o_dict:
    m_o = dict(m_o_dict)
print("Opened m_o")
with shelve.open("p_r_index.shelve")as p_r_dict:
    p_r = dict(p_r_dict)
print("Opened p_r")
with shelve.open("s_u_index.shelve") as s_u_dict:
    s_u = dict(s_u_dict)
print("Opened s_u")
with shelve.open("v_z_index.shelve") as v_z_dict:
    v_z = dict(v_z_dict)
print("Opened v_z")
with shelve.open("misc_index.shelve") as misc_dict:
    misc = dict(misc_dict)
print("Opened misc")

# with shelve.open('graph.shelve') as shelve_file:
#     adj_list = dict(shelve_file)
# print("Opened adj_list")
with shelve.open('pageRanks.shelve') as pageranks:
    pr_list = dict(pageranks)
print("Opened pr_list")
max_pr = max(pr_list.values())
min_pr = min(pr_list.values())
normalized_pr_list = {docid: (pr - min_pr) / (max_pr - min_pr) for docid, pr in pr_list.items()}
print("Normalized pageranks")

# def preprocess_adj_list(adj_list):
#     """
#     Convert outbound to inbound list.
#     """
#     inbound_links = defaultdict(list)
#     for page, outbound_links in adj_list.items():
#         for link in outbound_links:
#             inbound_links[link].append(page)
#     return inbound_links

# inbound_links = preprocess_adj_list(adj_list)
# print("Inbound links created")

def summarize(url):
    """
    Parse url for title and text content.
    Summarize with gpt-3.5 model.
    Return title and summary.
    """
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
        return None


@app.route('/')
def index():
    """
    Install html file.
    """
    return send_from_directory('.', 'index.html')


@app.route('/search')
def search():
    """
    Receive query items and documents.
    Search and time the query.
    Send summaries to frontend.
    """
    stemmer = SnowballStemmer("english")

    # parse query
    query = request.args.get('query')
    start_time = time.time_ns()
    query_lst = query.split()
    for i in range(len(query_lst)):
        query_lst[i] = stemmer.stem(query_lst[i].lower())

    # parse docs
    docs = dict()
    for q in query_lst:
        # docs[q] = data[q].copy()
        if 'a' <= q[0] <= 'c':
            docs[q] = a_c[q].copy()
        elif 'd' <= q[0] <= 'f':
            docs[q] = d_f[q].copy()
        elif 'g' <= q[0] <= 'i':
            docs[q] = g_i[q].copy()
        elif 'j' <= q[0] <= 'l':
            docs[q] = j_l[q].copy()
        elif 'm' <= q[0] <= 'o':
            docs[q] = m_o[q].copy()
        elif 'p' <= q[0] <= 'r':
            docs[q] = p_r[q].copy()
        elif 's' <= q[0] <= 'u':
            docs[q] = s_u[q].copy()
        elif 'v' <= q[0] <= 'z':
            docs[q] = v_z[q].copy()
        else:
            docs[q] = misc[q]

    results = cosine_sim(query_lst, docs)
    end_time = time.time_ns()
    duration = (end_time - start_time) / 1000000

    summaries = []
    for result in results:
        if summarize(result[0]) != None:
            title, summary = summarize(result[0])
            summaries.append({"title": title, "summary": summary, "url": result[0]})

    return jsonify(results=summaries, duration=duration)


def positions_check(query_items, match_list, max_distance=5):
    """
    Find all positions for each query item.
    Ensure item n positions are before item n+1.
    Returns set of valid docs that are close in proximity.
    """
    doc_positions = defaultdict(lambda: defaultdict(list))

    # collect positions
    for term in query_items:
        if term in match_list:
            postings = match_list[term]["postings"]
            for p in postings:
                doc_positions[p["docID"]][term].extend(p["positions"])

    valid_docs = set()

    # check positions
    for docid, positions in doc_positions.items():
        if all(term in positions for term in query_items):  # Ensure all terms are present in the doc
            valid = True
            previous_positions = sorted(positions[query_items[0]])
            for i in range(1, len(query_items)):
                current_positions = sorted(positions[query_items[i]])
                if not previous_positions or not current_positions:
                    valid = False
                    break
                # Find valid positions between previous and current term
                found = False
                j = 0
                for pos1 in previous_positions:
                    while j < len(current_positions) and current_positions[j] <= pos1:
                        j += 1
                    if j < len(current_positions) and current_positions[j] - pos1 <= max_distance:
                        found = True
                        previous_positions = current_positions[j:]
                        break
                if not found:
                    valid = False
                    break
            if valid:
                valid_docs.add(docid)

    return valid_docs


# def expand_root_set(root_set, adj_list, inbound_links):
#     base_set = set(root_set)
#     for page in root_set:
#         base_set.update(inbound_links[page])  # Include pages that link to pages in root set
#         for linked_page in inbound_links[page]:
#             base_set.update(adj_list[linked_page])  # Include pages that are linked to by pages in root set
#     return list(base_set)

# def normalize_scores(scores):
#     norm_factor = (sum(score ** 2 for score in scores.values())) ** 0.5
#     for page in scores:
#         scores[page] /= norm_factor

# def not_converged(old_hub_scores, hub_scores, old_auth_scores, auth_scores, tolerance):
#     hub_changes = sum(abs(old_hub_scores[page] - hub_scores[page]) for page in hub_scores)
#     auth_changes = sum(abs(old_auth_scores[page] - auth_scores[page]) for page in auth_scores)
#     return hub_changes > tolerance or auth_changes > tolerance

# def hits(relevant_docs, max_iterations=100, tolerance=0.0001):
#     base_set = expand_root_set(relevant_docs, adj_list, inbound_links)
#     hub_scores = {page: 1 for page in base_set}
#     auth_scores = {page: 1 for page in base_set}
    
#     for iteration in range(max_iterations):
#         old_hub_scores = hub_scores.copy()
#         old_auth_scores = auth_scores.copy()
        
#         # Authority update
#         for page in base_set:
#             if page not in inbound_links or page not in adj_list:
#                 continue
#             auth_scores[page] = sum(hub_scores[q] for q in inbound_links[page] if q in hub_scores)
#             hub_scores[page] = sum(auth_scores[q] for q in adj_list[page] if q in auth_scores)
        
#         # Normalization
#         normalize_scores(hub_scores)
#         normalize_scores(auth_scores)
        
#         # Check for convergence
#         if not not_converged(old_hub_scores, hub_scores, old_auth_scores, auth_scores, tolerance):
#             break
    
#     return hub_scores, auth_scores


def cosine_sim(query_items, match_list):
    """
    Return search results based on tf-idf score.
    """
    scores = defaultdict(float)
    doc_length = defaultdict(float)

    pr_weight = 0.75

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
                    # find the url's hit score in hit_list and add
                    pr_score = 0
                    if docid in pr_list:
                        pr_score = pr_list[docid]
                    # find the url's page rank score and add
                    scores[docid] += (1 - pr_weight) * (w_td * w_tq) + pr_weight * pr_score  #alter this line to consider pagerank, hub and authority scores maybe have to normalize
                    doc_length[docid] += (w_td * w_tq) ** 2 #(tf_idf ** 2) ** 0.5

    for d in scores:
        if doc_length[d] > 0:
            scores[d] /= doc_length[d]

    best_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return best_scores


if __name__ == "__main__":
    app.run()