from bs4 import BeautifulSoup
import os
import json
from collections import defaultdict
from nltk.stem.snowball import SnowballStemmer
import math
import hashlib
import numpy as np
import time
import shelve
import signal

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException

# Install timeout_handler to skip links with lengthy loads
signal.signal(signal.SIGALRM, timeout_handler)

def generate_fingerprint(weights):
    '''
    Generates a fingerprint for simhashing purposes.
    '''
    # initialize Vector V
    V = np.zeros(64, dtype=int)

    for word, weight in weights.items():
        # generate hash value
        hash_value = hashlib.sha256(word.encode()).digest()[:8]
        hash = ''.join(f"{byte:08b}" for byte in hash_value)

        # create vector V
        for i, bit in enumerate(hash):
            if bit == '1':
                V[i] += weight
            else:
                V[i] -= weight

    # generate fingerprint
    fingerprint = np.where(V > 0, 1, 0)
    return fingerprint

def wordFrequencies(text):
    '''
    Finds all the frequencies of words, accounts for ' and -.
    '''
    all_tokens = []
    token = ""
    stemmer = SnowballStemmer("english")
    positions_dict = defaultdict(list)
    pos = 0
    for c in text:
        if (('A' <= c <= 'Z') or ('a' <= c <= 'z') or ('0' <= c <= '9') or (c == "'") or (c == "-")):
            token += c
        else:
            if token:   # if token not empty, add token
                pos += 1
                all_tokens.append(stemmer.stem(token.lower()))
                positions_dict[stemmer.stem(token.lower())].append(pos)
                token = ""
    
    if token:   # add last token
        pos += 1
        all_tokens.append(stemmer.stem(token.lower()))
        positions_dict[stemmer.stem(token.lower())].append(pos)
    
    map = defaultdict(int)
    
    ### ATTEMPTED 2-gram and 3-gram indexing
    # loop through each token and increment its counter in the map
    # buff_str = []
    # buff_str_three = []
    for token in all_tokens:
        map[token] += 1
        # buff_str.append(token)
        # buff_str_three.append(token)
        # if(len(buff_str) == 2):
        #     map[" ".join(buff_str)] += 1
        #     for num in positions_dict[buff_str[0]]:
        #         positions_dict[" ".join(buff_str)].append(num)
        #     buff_str.pop(0)
        # if(len(buff_str_three) == 3):
        #     map[" ".join(buff_str_three)] += 1
        #     for num in positions_dict[buff_str_three[0]]:
        #         positions_dict[" ".join(buff_str_three)].append(num)
        #     buff_str_three.pop(0)

    return dict(map), dict(positions_dict)


def similarity(fingerprint1, fingerprint2):
    '''
    Compares 2 fingerprints and generate a similarity score.
    '''
    same_bits = sum(b1 == b2 for b1, b2 in zip(fingerprint1, fingerprint2))

    return same_bits / 64.0


def write_inverted_index(fingerprints, url, text, inverted_index, important_words, anchors):
    '''
    Write postings to local inverted_index.
    Posting contains these features: docID, tf, tf-idf, positions.
    '''
    frequencies, positions = wordFrequencies(text)
    
    fingerprint = generate_fingerprint(frequencies)
    found = 0
    for prevFingerprint in fingerprints:
        if similarity(fingerprint, prevFingerprint) >= (60.8/64):   # 95% similarity
            found = 1
            break
    if found:
        return fingerprint, 0
    
    important_frequencies, _ = wordFrequencies(important_words)

    for token, frequency in frequencies.items():
        add_count = frequency
        pos_list = positions[token]
        if token in important_frequencies:
            add_count += important_frequencies[token] * 2
        if token in inverted_index:
            inverted_index[token]["df"] += 1
            inverted_index[token]["postings"].append({"docID": url, "tf": add_count, "tf-idf": None, "positions": pos_list})
        else:
            inverted_index[token] = {"df": 1,"idf": None,"postings": [{"docID": url, "tf": add_count, "tf-idf": None, "positions": pos_list}]}
    
    for anchor_text, anchor_url in anchors:
        anchor_freqs, anchor_pos = wordFrequencies(anchor_text)
        for token, frequency in anchor_freqs.items():
            pos_list = anchor_pos[token]
            if token in inverted_index:
                inverted_index[token]["df"] += 1
                inverted_index[token]["postings"].append({"docID": anchor_url, "tf": frequency, "tf-idf": None, "positions": pos_list})
            else:
                inverted_index[token] = {"df": 1,"idf": None,"postings": [{"docID": anchor_url, "tf": frequency, "tf-idf": None, "positions": pos_list}]}
    
    return fingerprint, 1
        

def write_file(inverted_index, file_name):
    '''
    Write local inverted_index to json file.
    '''
    print(file_name)
    
    old_data = shelve.open(file_name)
    
    for token, info in inverted_index.items():
        old_data[token] = info
        old_data.sync()
    
    old_data.close()


def merge_postings(existing_postings, new_postings):
    """
    Merge new postings into current postings.
    """
    return existing_postings + new_postings


def merge_terms(term1, term2):
    """
    Merge frequency and postings of two terms.
    """
    term1['df'] += term2['df']
    term1['postings'] = merge_postings(term1['postings'], term2['postings'])
    term1['idf'] = None  # idf will be set to None as specified
    return term1


def merge_all_files(dump_count):
    """
    Merge multiple shelve files into single shelve file.
    """
    shelve_files = [f'inverted_index_{i}.shelve' for i in range(1, dump_count + 1)]

    with shelve.open('inverted_index_total.shelve', 'c') as index:
        for shelve_file in shelve_files:
            with shelve.open(shelve_file, 'r') as partial:
                for key in partial:
                    if key in index:
                        index[key] = merge_terms(index[key], partial[key])
                        index.sync()
                    else:
                        index[key] = partial[key].copy()
                        index.sync()


def fill_and_split(total_page_count):
    """
    Create tf-idf value for each posting.
    """
    a_c = shelve.open("a_c_index.shelve")
    d_f = shelve.open("d_f_index.shelve")
    g_i = shelve.open("g_i_index.shelve")
    j_l = shelve.open("j_l_index.shelve")
    m_o = shelve.open("m_o_index.shelve")
    p_r = shelve.open("p_r_index.shelve")
    s_u = shelve.open("s_u_index.shelve")
    v_z = shelve.open("v_z_index.shelve")
    misc = shelve.open("misc_index.shelve")
    
    #Filling in the appropriate tf-idf values in index
    with shelve.open("inverted_index_total.shelve") as index:
        for key, value in index.items():
            value["idf"] = math.log10(total_page_count/value["df"])
            for posting in value["postings"]:
                posting["tf-idf"] = value["idf"] * posting["tf"]
            index[key] = value
            index.sync()
            
            if 'a' <= key[0] <= 'c':
                a_c[key] = value
                a_c.sync()
            elif 'd' <= key[0] <= 'f':
                d_f[key] = value
                d_f.sync()
            elif 'g' <= key[0] <= 'i':
                g_i[key] = value
                g_i.sync()
            elif 'j' <= key[0] <= 'l':
                j_l[key] = value
                j_l.sync()
            elif 'm' <= key[0] <= 'o':
                m_o[key] = value
                m_o.sync()
            elif 'p' <= key[0] <= 'r':
                p_r[key] = value
                p_r.sync()
            elif 's' <= key[0] <= 'u':
                s_u[key] = value
                s_u.sync()
            elif 'v' <= key[0] <= 'z':
                v_z[key] = value
                v_z.sync()
            else:
                misc[key] = value
                misc.sync()
    
    a_c.close()
    d_f.close()
    g_i.close()
    j_l.close()
    m_o.close()
    p_r.close()
    s_u.close()
    v_z.close()
    misc.close()

        
def indexer(path):
    """
    Build inverted index.
    Dump to disk routinely.
    Implements anchor words, PageRank algorithm, and HITS algorithm.
    """
    if not os.path.exists(path):
        print("Directory not found")
        return
    
    # inverted index structure
    # {token: [{url: frequency}, {...}], token: ...}
    inverted_index = {}
    adj_matrix = defaultdict(list)
    
    # Counts to keep track of things
    page_count = 0
    dump_count = 0
    total_page_count = 0
    fingerprints = []
    # loop through each folder in DEV
    for domain in os.listdir(path):

        # #LIMITING ROUNDS FOR TESTING, DELETE!!!!!!!!!!!
        # if total_page_count > 100:
        #     break
        # #LIMITING ROUNDS FOR TESTING, DELETE!!!!!!!!!!!

        domain_path = os.path.join(path, domain)
        # loop through each json & extract content using encoding
        for page in os.listdir(domain_path):

            # #LIMITING ROUNDS FOR TESTING, DELETE!!!!!!!!!!!
            # if total_page_count > 100:
            #     break
            # #LIMITING ROUNDS FOR TESTING, DELETE!!!!!!!!!!!

            page_path = os.path.join(domain_path, page)
            print(page_path, page_count)
            try:
                signal.alarm(5)
                with open(page_path, 'r') as json_file:
                    # Accessing page
                    data = json.load(json_file)
                    url = data["url"]
                    content = data["content"]
                    soup = BeautifulSoup(content, 'html.parser')
                    text = soup.get_text()
                    
                    # Getting important words
                    important = soup.find_all(['b','strong','h1','h2','h3','title'])
                    combined_important = ' '.join(x.get_text() for x in important)

                    # Anchor words
                    anchor_tags = soup.find_all('a')
                    anchors = []
                    for tag in anchor_tags:
                        href = tag.get("href")
                        if href != "#" and href is not None:
                            anchors.append((tag.get_text(),href))
                            # Page Rank implementation
                            adj_matrix[url].append(href)
                            
                    # Writing to index in memory
                    fingerprint, code = write_inverted_index(fingerprints, url, text, inverted_index, combined_important, anchors)
                    fingerprints.append(fingerprint)
                    if len(fingerprints) > 15:
                        fingerprints.pop(0)
                    if code:
                        total_page_count += 1
                        
                    # Incrementing counts
                    page_count += 1

                    signal.alarm(0)
                    
                    # Dump to disk every 10000 pages processed
                    if page_count > 10000: #TESTING CHANGED TO 10, CHANGE BACK!!!!!!!!!!!!!
                        dump_count += 1
                        write_file(inverted_index, f"inverted_index_{dump_count}.shelve")
                        inverted_index = defaultdict(list)
                        page_count = 0
            except TimeoutException:
                print(f"Skipping {page_path} due to timeout")
                signal.alarm(0)
                continue
            except json.JSONDecodeError as e:
                print("Error parsing JSON file:", str(e))
            finally:
                signal.alarm(0)

    #Final dump
    dump_count += 1
    write_file(inverted_index, f"inverted_index_{dump_count}.shelve")
    
    write_file(adj_matrix, "graph.shelve")
    
    return dump_count, total_page_count


def preprocess_adj_list(adj_list):
    """
    Convert outbound to inbound list.
    """
    inbound_links = defaultdict(list)
    for page, outbound_links in adj_list.items():
        for link in outbound_links:
            inbound_links[link].append(page)
    return inbound_links


def hits(query):
    """
    Implement HITS algorithm.
    """
    with shelve.open('graph.shelve') as shelve_file:
        adj_list = dict(shelve_file)
    inbound_links = preprocess_adj_list(adj_list)
    result_dict = dict()
    # based on the query
    relevant_url_list = set()
    for q in query:
        if q in adj_list:
            for item in adj_list[q]:
                relevant_url_list.add(item)
        if q in inbound_links:
            for item in inbound_links[q]:
                relevant_url_list.add(item)
    # have a list of relevant url in relevant_url_list
    # find all the url in shelve, if url in relevant, then we add the url to the dict and have value as tuple (hub, auth)
    #for relevant_url in relevant_url_list:
    return None
    # read the urls within the match_list, find url that connect to these and place them into the match_list
    # rank based on 


def pagerank(d=0.85, max_iterations=100, tolerance=1.0e-6):

    """
    Implement PageRank algorithm.
    """
    with shelve.open('graph.shelve') as shelve_file:
        adj_list = dict(shelve_file)
    
    inbound_links = preprocess_adj_list(adj_list)
    pages = list(adj_list.keys())
    N = len(pages)
    pageRank = {page: 1.0 / N for page in pages}
    newRank = pageRank.copy()
    
    for iteration in range(max_iterations):
        print(f"Iteration: {iteration}")
        for page in pages:
            sum_rank = 0
            for linking_page in inbound_links[page]:
                sum_rank += pageRank[linking_page] / len(adj_list[linking_page])
            newRank[page] = (1 - d) / N + d * sum_rank
        
        # Check for convergence
        if all(abs(newRank[page] - pageRank[page]) < tolerance for page in pages):
            print(f"Converged after {iteration + 1} iterations")
            break
        
        # Update pageRank values
        pageRank.update(newRank)
    
    return pageRank

if __name__ == "__main__":
    # start_time = time.process_time_ns()
    # dump_count, total_page_count = indexer("DEV")
    # merge_all_files(dump_count)
    # fill_and_split(total_page_count)
    # end_time = time.process_time_ns()
    # with open("time.txt", 'w') as file:
    #     file.write(f"Indexing time: {end_time - start_time}")
    # pageRanks = pagerank()
    # write_file(pageRanks, "pageRanks.shelve")
    count = 0
    with shelve.open('pageRanks.shelve') as db:
        for key in db:
            count += 1
        print(count)