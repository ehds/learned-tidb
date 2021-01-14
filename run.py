from connect.connect import DBConnect
from utils.file_helper import write_json
from utils.join_order import extract_join_tree
db = DBConnect('127.0.0.1', 'root', '', 'test', 4000)
data = db.analyze(
    'select * from A,B where A.id = B.id and A.name > 2 and A.name <5 and A.name like "%ab"')
# data = db.analyze('select * from A left join B on A.id = B.id and A.name = B.name')
# data = db.analyze('select * from A')
write_json('data/test.json', data)
a = extract_join_tree('data/test.json')
print(a)
