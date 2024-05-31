import shelve
from nltk.stem.snowball import SnowballStemmer

if __name__ == "__main__":
    stemmer = SnowballStemmer("english")
    data = shelve.open("inverted_index.shelve")
    while True:
        query = input("What do you want to search?: ")
        query = query.split()
        docs = list()
        for token in query:
            docs.append(sorted(data[stemmer.stem(token.lower())], key=lambda x: list(x.keys())[0]))
        sorted_lists = sorted(docs, key=lambda x: len(x))

        if len(sorted_lists) > 1:
            matches = sorted_lists[0]
            temp = list()
            for i in range(1,len(sorted_lists)):
                #Comparing sorted_lists = [[{},{},{}],[{},{}],[]]
                j, k = 0, 0
                while j<len(matches) and k<len(sorted_lists[i]):
                    if list(matches[j].keys())[0] == list(sorted_lists[i][k].keys())[0]:
                        temp.append(matches[j])
                        j += 1
                        k += 1
                    elif list(matches[j].keys())[0] < list(sorted_lists[i][k].keys())[0]:
                        j += 1
                    else:
                        k += 1
                matches = temp
                temp = list()
            if len(matches) > 5:
                for posting in matches[:5]:
                    print(list(posting.keys())[0])
            else:
                for posting in matches:
                    print(list(posting.keys())[0])
        else:
            if len(sorted_lists[0]) > 5:
                for posting in sorted_lists[0][:5]:
                    print(list(posting.keys())[0])
            else:
                for posting in sorted_lists[0]:
                    print(list(posting.keys())[0])
                
        # here cause aud was being dum
        #sorted_lists = [ [ {} ], [ {}, {} ], [ {}, {}, {} ] ]
        #sorted_lists[2] = [ {}, {}, {} ]
        #sorted_lists[2][0] = {}
        #sorted_lists[2][0].keys() = dict_keys(singular url)