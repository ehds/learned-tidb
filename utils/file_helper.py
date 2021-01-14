import json


def write_json(path, data, encoding='utf-8'):
    with open(path, 'w', encoding=encoding) as f:
        json.dump(data, f, ensure_ascii=False)
