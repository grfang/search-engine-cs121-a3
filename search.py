import os
import shelve

if __name__ == "__main__":
    data = shelve.open("inverted_index_55393.shelve")
    while True:
        query = input("What do you want to search?")
        query = query.split()
        pages = data[query[0]]
        sorted_pages = sorted(pages, key=lambda x: list(x.values())[0], reverse=True)
        for i in range(5):
            print(sorted_pages[i])