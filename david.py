import shelve

if __name__ == "__main__":
    with shelve.open('inverted_index_test.shelve') as db:
        for key in db:
            print(key, db[key])

    db.close()
