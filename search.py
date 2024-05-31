import shelve
from nltk.stem.snowball import SnowballStemmer
from collections import defaultdict
import math


def cosine_sim(query, match_list):
    query_items = query.lower().split()
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
    stemmer = SnowballStemmer("english")
    data = shelve.open("inverted_index_test.shelve")
    data.close()