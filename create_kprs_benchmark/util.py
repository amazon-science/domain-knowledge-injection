# General helper functions

import json
import sqlite3


def read_jsonl(file_path):
    """ Reads a .jsonl file. """
    records = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            records.append(json.loads(line))
    return records


def load_mwoz_databases(database_dir, all_json=True):
    """ Helper function for loading the MultiWOZ databases (adapted from the MultiWOZ codebase) """
    # 'police' and 'taxi' domains are excluded due to simplicity
    domains = ['restaurant', 'hotel', 'attraction', 'train', 'hospital', 'bus', 'police']
    dbs = {}
    for domain in domains:
        if domain in ['hospital', 'police'] or all_json:
            db = '{}/{}_db.json'.format(database_dir, domain)
            with open(db, 'r', encoding='utf8') as in_f:
                dbs[domain] = json.load(in_f)
        else:
            db = '{}/{}-dbase.db'.format(database_dir, domain)
            conn = sqlite3.connect(db)
            c = conn.cursor()
            dbs[domain] = c

            # Inspect database
            print('-' * 10)
            print('Database for the {} domain:'.format(domain))
            dbs[domain].execute('SELECT name FROM sqlite_master WHERE type=\'table\';')
            print('Tables: ', dbs[domain].fetchall())
            dbs[domain].execute('SELECT * FROM {}'.format(domain))
            print(dbs[domain].description)
            rows = dbs[domain].fetchall()
            for row in rows[:10]:
                print(row)
    return dbs
