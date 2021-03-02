import torch
from torch.nn import init
import torchfold
import torch.nn as nn
from utils.join_order import JoinPlan, TableReader
import torch.nn.functional as F
from learnedjoin.DQN.graph_conv import GraphConvolution
import numpy as np


class TreeLSTM(nn.Module):
    def __init__(self, num_units):
        super(TreeLSTM, self).__init__()
        self.num_units = num_units
        self.FC1 = nn.Linear(num_units, 5 * num_units)
        self.FC2 = nn.Linear(num_units, 5 * num_units)
        self.FC0 = nn.Linear(num_units, 5 * num_units)
        self.LNh = nn.LayerNorm(num_units,)
        self.LNc = nn.LayerNorm(num_units,)

    def forward(self, left_in, right_in, inputX):
        lstm_in = self.FC1(left_in[0])
        lstm_in += self.FC2(right_in[0])
        lstm_in += self.FC0(inputX)
        a, i, f1, f2, o = lstm_in.chunk(5, 1)
        c = (a.tanh() * i.sigmoid() + f1.sigmoid() * left_in[1] +
             f2.sigmoid() * right_in[1])
        h = o.sigmoid() * c.tanh()
        return h, c


class TreeRoot(nn.Module):
    def __init__(self, num_units):
        super(TreeRoot, self).__init__()
        self.num_units = num_units
        self.FC = nn.Linear(num_units, num_units)
        self.sum_pooling = nn.AdaptiveAvgPool2d((1, num_units))
        # self.max_pooling = nn.AdaptiveAvgPool2d((1,num_units))
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, tree_list):

        return self.relu(self.FC(self.sum_pooling(tree_list)).view(-1, self.num_units))


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, ofeat, dropout=0.8):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, ofeat)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        x = F.relu(x)
        return x
        # return F.log_softmax(x, dim=1)


class TreeTLSTMQFunction(nn.Module):
    def __init__(self, size, all_tables, all_colums, device):
        super(TreeTLSTMQFunction, self).__init__()
        self.size = size
        self.all_tables = all_tables
        self.all_columns = all_colums
        self.column_num = len(all_colums)
        self.table_num = len(all_tables)
        # self.column_embeddings = nn.Embedding(column_num, size)
        self.table_embedings = nn.Embedding(self.table_num, size)
        self.leafLn = nn.LayerNorm(size,)
        self.join_condition_fc = nn.Linear(self.column_num, size)
        self.tree_lstm = TreeLSTM(self.size)
        self.tree_root = TreeRoot(self.size)
        self.out = nn.Linear(size, 1)
        self.fc = nn.Linear(3*size, size)
        # 1: join_vector size
        self.gcn = GCN(1, 2*size, size, 0.8)
        self.gcn_fc = nn.Linear(108, 1)
        self.device = device

    def table2index(self, table):
        for i in range(len(self.all_tables)):
            if table in self.all_tables[i]:
                return i

    def leaf(self, table_id):
        return self.leafLn(self.table_embedings(table_id)), torch.zeros((table_id.shape[0], self.size), device=self.device)

    def join_conditions(self, equal_join_columns):
        join_columns = torch.zeros((self.column_num), device=self.device)
        join_columns[equal_join_columns] = 1
        output = self.join_condition_fc(join_columns)
        return output

    def encoding_tree(self, join_tree: JoinPlan):
        # join node
        if type(join_tree) == JoinPlan:
            left_h, left_c = self.encoding_tree(join_tree.left_node)
            right_h, right_c = self.encoding_tree(join_tree.right_node)
            join_columns = []
            for condition in join_tree.conditions:
                if condition.function != 'eq':
                    continue
                for column in condition.args:
                    join_columns.append(self.all_columns.index(column))
            inputx = self.join_conditions(join_columns)
            return self.tree_lstm((left_h, left_c), (right_h, right_c), inputx)
        # table reader
        else:
            assert type(join_tree) == TableReader
            for index in range(self.table_num):
                if join_tree.table in self.all_tables[index]:
                    table_id = torch.LongTensor([index]).to(self.device)
                    return self.leaf(table_id)

    def forward(self, state, action):
        state_encodings = torch.cat(
            [self.encoding_tree(s.join_tree)[0] for s in state], dim=0)

        # get origianl tree encoding
        original_trees = [s.origin_tree.encoding_join_conditions()
                          for s in state]
        adjs = torch.FloatTensor(
            np.array([t[0] for t in original_trees])).to(self.device)
        join_vectors = torch.FloatTensor(
            np.array([t[1] for t in original_trees])).to(self.device)
        original_tree_encodings = self.gcn(join_vectors, adjs).transpose(2, 1)
        original_tree_encodings = self.gcn_fc(
            original_tree_encodings).squeeze(2)

        action_encoding = []
        for a in action:
            if type(a.join_table) == TableReader:
                x = self.leaf(torch.LongTensor(
                    [self.table2index(a.join_table.table)]).to(self.device))[0]
                action_encoding.append(x)
            else:
                action_encoding.append(self.encoding_tree(a.join_table)[0])
        action_encoding = torch.cat(action_encoding, dim=0)
        x = torch.cat([state_encodings, action_encoding,
                       original_tree_encodings], dim=1)
        inputx = self.fc(x)
        return self.out(F.relu(inputx))
        # return self.tree_root([state_encodings, action_encoding])
