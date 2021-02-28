# -*- coding:utf-8 -*-
import json
import pickle


def write_json(path, data, encoding='utf-8'):
    with open(path, 'w', encoding=encoding) as f:
        json.dump(data, f, ensure_ascii=False)


def write_db_info(data, database):
    with open(f'data/{database}.pkl', 'wb') as f:
        pickle.dump(data, f)


def get_db_info(database):
    data = None
    with open(f'data/{database}.pkl', 'rb') as f:
        data = pickle.load(f)
    return data
