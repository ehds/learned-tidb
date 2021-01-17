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
# 'select * from A left join B on (A.id = B.id and A.name = B.name) ')
# data = db.analyze('select * from A where A.id >1 and A.name =4 and A.id < 4')
# write_json('data/test.json', data)
a = extract_join_tree('data/test.json')
b = convert_tree_to_trajectory(a)
tree, indexes = a.encode()

net = nn.Sequential(
    tcnn.BinaryTreeConv(4, 16),
    tcnn.TreeLayerNorm(),
    tcnn.TreeActivation(nn.ReLU()),
    tcnn.BinaryTreeConv(16, 8),
    tcnn.TreeLayerNorm(),
    tcnn.TreeActivation(nn.ReLU()),
    tcnn.BinaryTreeConv(8, 4),
    tcnn.TreeLayerNorm(),
    tcnn.TreeActivation(nn.ReLU()),
    tcnn.DynamicPooling()
)

trees = torch.Tensor(np.array([tree]))
indexes = torch.LongTensor(np.array([indexes]))
print(trees, indexes)
print(trees.shape, indexes.shape)
print(net((trees, indexes)).shape)
# [eq:['test.a.id', ' test.b.id'], eq:['test.a.name', ' test.b.name']]
# [or:['and(eq(test.a.id', ' test.b.id'], eq:['test.a.name', ' test.b.name'], gt:['test.a.id', ' 5']]
