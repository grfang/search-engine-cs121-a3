import shelve

if __name__ == "__main__":
    with shelve.open('a_c_index.shelve') as db:
        for key in db:
            print(key, db[key])

    db.close()
