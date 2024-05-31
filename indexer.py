from bs4 import BeautifulSoup
import os
import json
from collections import defaultdict
from nltk.stem.snowball import SnowballStemmer
import math

import shelve

def wordFrequencies(text):
    '''
    Finds all the frequencies of words, accounts for ' and -
    '''
    all_tokens = []
    token = ""
    stemmer = SnowballStemmer("english")
    for c in text:
        if (('A' <= c <= 'Z') or ('a' <= c <= 'z') or ('0' <= c <= '9') or (c == "'") or (c == "-")):
            token += c
        else:
            if token:   # if token not empty, add token
                all_tokens.append(stemmer.stem(token.lower()))
                token = ""
    
    if token:   # add last token
        all_tokens.append(stemmer.stem(token.lower()))
    
    map = defaultdict(int)
    
    # loop through each token and increment its counter in the map
    for token in all_tokens:
        map[token] += 1
    
    return dict(map)


def write_inverted_index(url, text, inverted_index, important_words):
    '''
    Write postings to local inverted_index
    '''
    frequencies = wordFrequencies(text)
    important_frequencies = wordFrequencies(important_words)

    for token, frequency in frequencies.items():
        add_count = frequency
        if token in important_frequencies:
            add_count += important_frequencies[token] * 2
        if token in inverted_index:
            inverted_index[token]["df"] += 1
            inverted_index[token]["postings"].append({"docID": url, "tf": add_count, "tf-idf": None})
        else:
            inverted_index[token] = {"df": 1,"idf": None,"postings": [{"docID": url, "tf": add_count, "tf-idf": None}]}
        

def write_file(inverted_index, file_name):
    '''
    Write local inverted_index to json file
    '''
    
    old_data = shelve.open(file_name)
    
    for token, info in inverted_index.items():
        old_data[token] = info
        old_data.sync()
    
    old_data.close()


def merge_files(f1, f2): 
    with open(f1, 'r') as f1, open(f2, 'r') as f2:
        while True:
            line1 = f1.readline()
            line2 = f2.readline()
            if not f1 and not f2:
                break
            
            if not f1 and f2:
                print("F1 finished, we should copy everything in f2 to the new json file")
            
            if not f2 and f1:
                print("F1 finished, we should copy everything in f2 to the new json file")
            
            else:
                #Compare line1 and line2. 
                #If they are similar, then we merge them and put them into a json file
                #else, we write which ever higher in alphabetical order to the json file then move to the next line
                continue
                
 
        
def indexer(path):
    if not os.path.exists(path):
        print("Directory not found")
        return
    
    # inverted index structure
    # {token: [{url: frequency}, {...}], token: ...}
    inverted_index = {}
    
    # TODO: every 10k files, write, then reset inverted_index
    page_count = 0
    total_page_count = 0
    # loop through each folder in DEV
    for domain in os.listdir(path):

        #LIMITING ROUNDS FOR TESTING, DELETE!!!!!!!!!!!
        if total_page_count > 30:
            break
        #LIMITING ROUNDS FOR TESTING, DELETE!!!!!!!!!!!

        domain_path = os.path.join(path, domain)
        # loop through each json & extract content using encoding
        for page in os.listdir(domain_path):

            #LIMITING ROUNDS FOR TESTING, DELETE!!!!!!!!!!!
            if total_page_count > 30:
                break
            #LIMITING ROUNDS FOR TESTING, DELETE!!!!!!!!!!!

            page_path = os.path.join(domain_path, page)
            print(page_path)
            with open(page_path, 'r') as json_file:
                try:
                    data = json.load(json_file)
                    url = data["url"]
                    content = data["content"]
                    soup = BeautifulSoup(content, 'html.parser')
                    text = soup.get_text()
                    important = soup.find_all(['b','strong','h1','h2','h3','title'])
                    combined_important = ' '.join(x.get_text() for x in important)
                    write_inverted_index(url, text, inverted_index, combined_important)
                    # page_count += 1
                    total_page_count += 1
                    # if page_count > 10000:
                    #     write_file(inverted_index, f"inverted_index_{total_page_count}.json")
                    #     inverted_index = defaultdict(list)
                    #     page_count = 0
                except json.JSONDecodeError as e:
                    print("Error parsing JSON file:", str(e))

    for value in inverted_index.values():
        value["idf"] = math.log10(total_page_count/value["df"])
        for posting in value["postings"]:
            posting["tf-idf"] = value["idf"] * posting["tf"]

    # print(inverted_index)
    write_file(inverted_index, f"inverted_index_test.shelve")


if __name__ == "__main__":
    indexer("DEV")

    # #TODO: merge every pair of files
    #merge_files()