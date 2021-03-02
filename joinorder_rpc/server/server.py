from concurrent import futures
import grpc
from . import join_order_pb2
from . import join_order_pb2_grpc
from .logical_plan import encode_logical_plan
from learnedjoin.DQN.dqn import JoinOrderDQN, TreeCNNQFunction
import torch
import numpy as np
from utils.join_order import State, Action, extract_join_tree


class Greeter(join_order_pb2_grpc.GreeterServicer):

    def SayHello(self, request, context):
        print(request.name)
        return join_order_pb2.HelloReply(message='Hello, %s!' % request.name)


def test_model(state, action):
    r""" state,action : batch * in_channel * node_size """
    obs_dim = state[0].shape[0]
    action_dim = action[0].shape[0]
    state = (torch.Tensor([state[0]]), torch.LongTensor([state[1]]))
    action = (torch.Tensor([action[0]]), torch.LongTensor([action[1]]))
    q_net = TreeCNNQFunction(obs_dim, 30, action_dim)
    return q_net(state, action).flatten()  # batch * 1 q_value


def get_best_action(model, state, actions):
    r""" Get  lowest cost action of actions with current state"""
    # TODO batch feed to Q net
    costs = []
    # get encoding of state and actions
    current_join_tree = torch.Tensor([state[0]])
    tree_index = torch.LongTensor([state[1]])
    for action in actions:
        action_tree = torch.Tensor([action[0]])
        action_index = torch.LongTensor([action[1]])
        with torch.no_grad():
            q_value = model((current_join_tree, tree_index),
                            (action_tree, action_index))
            costs.append(q_value.flatten().numpy()[0])
    return np.argmin(costs)


class JoinOrder(join_order_pb2_grpc.JoinOrderServicer):
    def __init__(self, join_order_model, database, use_dqn=False):
        self.model = join_order_model
        self.use_dqn = use_dqn
        self.database = database
        self.set_is_dqn()

    def set_is_dqn(self):
        with open('config.conf', 'w') as f:
            f.write(str(self.use_dqn))

    def TestJoinNode(self, logical_node, context):
        for item in logical_node.conditions:
            print(item.funcname, item.args)
        return join_order_pb2.HelloReply(message='Hello')

    # control remote tidb if use dqn or keep origin
    def IsDQN(self, empty, context):
        flag = False
        with open('config.conf', 'r') as f:
            value = f.readline()
            flag = (value == 'True')
        return join_order_pb2.BoolValue(value=flag)

    def _get_join_tree_from_sql(self, sql):
        # TODO gracful
        origin_is_dqn = self.use_dqn
        self.use_dqn = False
        self.set_is_dqn()

        sql = sql.replace('explain', '')
        sql = sql.replace('analyze', '')
        explain_info = self.database.explain(sql)
        assert explain_info != None
        join_tree = extract_join_tree(explain_info)

        self.use_dqn = origin_is_dqn
        self.set_is_dqn()
        return join_tree

    def GetAction(self, state, context):
        # print(state.current_join_tree.tp)
        best_action_index = 0
        origial_sql = state.original_sql
        origial_tree = self._get_join_tree_from_sql(origial_sql)
        logical_plan = encode_logical_plan(
            state.current_join_tree)
        encode_actions = [Action(encode_logical_plan(
            action), []) for action in state.actions]
        state = State([], logical_plan, [], origial_tree)
        best_action_index = self.model.get_best_action(
            state, encode_actions)
        # except Exception as e:
        #     print(f"get action error {e}")
        return join_order_pb2.HelloReply(message=f'{best_action_index}')


def serve(model, db, use_dqn):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    join_order_pb2_grpc.add_GreeterServicer_to_server(Greeter(), server)
    join_order_pb2_grpc.add_JoinOrderServicer_to_server(
        JoinOrder(model, db, use_dqn), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()
