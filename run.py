from learnedjoin.DQN.dqn import JoinOrderDQN
from connect.connect import DBConnect
from utils.file_helper import write_json
from utils.join_order import extract_join_tree, convert_tree_to_trajectory
import numpy as np
import learnedjoin.DQN.tcnn as tcnn
import torch
import torch.nn as nn
# db = DBConnect('127.0.0.1', 'root', '', 'test', 4000)
# data = db.analyze(
# 'select * from A,B where A.id = B.id and A.name > 2 and A.name <5 and A.name like "%ab"')
# data = db.analyze(
#     'select * from A,B,C where A.id = B.id and B.name = C.name')
# data = db.analyze('select * from A where A.id >1 and A.name =4 and A.id < 4')
# write_json('data/test2.json', data)
a = extract_join_tree('data/test2.json')
b = convert_tree_to_trajectory(a)
join_dqn = JoinOrderDQN(10, 1, 4, 4)
for i in range(1, len(b)-1):
    cur = b[i]
    next = b[i+1]
    join_dqn.replay_buffer.add_observation(
        cur.state.encode(), cur.action.encode(), cur.reward, next.state.encode(), next.state.is_done)
join_dqn.train()
tree, indexes = a.encode()
# join_order_dqn = JoinOrderDQN(10, 2, 4, 4)

# [eq:['test.a.id', ' test.b.id'], eq:['test.a.name', ' test.b.name']]
# [or:['and(eq(test.a.id', ' test.b.id'], eq:['test.a.name', ' test.b.name'], gt:['test.a.id', ' 5']]
