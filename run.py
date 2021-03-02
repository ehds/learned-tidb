# -*- coding:utf-8 -*-
from learnedjoin.DQN.dqn import JoinOrderDQN
from database.connect import DB
from utils.file_helper import write_json
from utils.join_order import extract_join_tree_from_path, convert_tree_to_trajectory
import numpy as np
import learnedjoin.DQN.tcnn as tcnn
import torch
import torch.nn as nn
import joinorder_rpc.server.server as server
import logging
import threading
import time
import os
from database.workload import WorkLoad


def train_model(model: JoinOrderDQN):
    print("training")
    # wating rpc server is running
    time.sleep(2)
    model.train()


if __name__ == '__main__':
    logging.basicConfig()
    workload = WorkLoad(
        "./join-order-benchmark")

    db = DB('127.0.0.1', 'root', '', 'imdb', 4000)
    db_for_server = DB('127.0.0.1', 'root', '', 'imdb', 4000)
    print(len(db.unique_columns))
    a = extract_join_tree_from_path(os.path.join("data", "33c.json"))
    print(a.conditions)
    # a.encode()
    b = convert_tree_to_trajectory(a)
    obs_dim = b[0].state.encode()[0].shape[0]
    act_dim = b[0].action.encode()[0].shape[0]
    model = JoinOrderDQN(db, workload, 32, 1, obs_dim, act_dim, False)
    threading.Thread(target=train_model, args=(model,)).start()
    server.serve(model, db_for_server, True)
