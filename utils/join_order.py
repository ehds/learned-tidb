import json
from enum import Enum
import re
import math
import queue
import copy
import numpy as np


class Plan():
    r""" Abstract class for logical plan and physical plan """

    def __init__(self):
        self.id = 0
        # -1 means nothing
        self._execute_time = None  # actual exectution time (ms)
        self.cost = None          # estimate cost
        self.actual_row = None    # actual rows
        self.est_rows = None      # estimate rows
        self.children = []        # children plans

    def encode(self):
        raise NotImplementedError()

    @property
    def execute_time(self):
        return self._execute_time

    @execute_time.setter
    def execute_time(self, value):
        r""" Unit conversion (ms)
        reference: tidb/util/execdetails/execdetails.go#FormatDuration
        [s,ms,us,ns]
        """
        units = ['ns', 'µs', 'ms', 's']
        unified_unit_index = 2  # ms

        # match time string format: 22.22us
        match_time = re.search(r'(\d*[\.]*\d*)(\w+)', value)
        execution_time = float(match_time.group(1))
        time_unit = match_time.group(2)
        assert time_unit in units, f"time unit {time_unit} not existis"
        time_unit_index = units.index(time_unit)

        # unfied unit and rounded to two decimals
        unfied_execute_time = execution_time * \
            math.pow(10, (time_unit_index-unified_unit_index)*3)
        self._execute_time = round(unfied_execute_time, 2)


class JoinPlan(Plan):
    def __init__(self, join_type, conditions):
        super(JoinPlan, self).__init__()
        self.join_type = join_type
        self.conditions = conditions
        self.left_node = None
        self.right_node = None

    def get_features(self):
        return np.array([1, 2, 3, 4])

    def encode(self):
        # raise NotImplementedError()
        return self.preorder_encode()

    def preorder_encode(self):
        r""" TreeCNN need preorder encoding """
        preorder = []

        def recurse(x, idx):
            preorder.append(x.get_features())
            tree_cnn_indexes = []
            # leaf node
            if type(x) == TableReader:
                return [[idx, 0, 0]]
            left_tree_cnn_indexes = recurse(x.left_node, idx+1)
            left_tree_count = len(preorder)
            right_tree_cnn_indexes = recurse(x.right_node, left_tree_count+1)
            tree_cnn_indexes = [[
                idx, left_tree_cnn_indexes[0][0], right_tree_cnn_indexes[0][0]]]
            tree_cnn_indexes.extend(left_tree_cnn_indexes)
            tree_cnn_indexes.extend(right_tree_cnn_indexes)
            return tree_cnn_indexes

        tree_cnn_indexes = recurse(self, 1)
        preorder = [[0, 0, 0, 0]]+preorder
        return np.array(preorder).transpose(1, 0), np.array(tree_cnn_indexes).flatten().reshape(-1, 1)

    def __str__(self):
        return f"{self.join_type},time:{self.execute_time},left:{self.left_node},right:{self.right_node}"

    def __repr__(self):
        return self.__str__()


class TableReader(Plan):
    def __init__(self, table='', conditions=[]):
        self.table = table
        self.conditions = conditions

    def get_features(self):
        return np.array([1, 1, 1, 1])

    def encode(self):
        return self.get_features()

    def __str__(self):
        return f"reader:{self.table},time:{self.execute_time}"

    def __repr__(self):
        return self.__str__()


class Condition():
    def __init__(self, function, args):
        self.function = function
        self.args = args

    def encode(self):
        pass

    def __str__(self):
        return f"{self.function}:{self.args}"

    def __repr__(self):
        return self.__str__()
        # JoinType = Enum('InnerJoin', 'LeftOuterJoin', 'RightOuterJoin',
        #                 'SemiJoin', 'AntiSemiJoin', 'LeftOuterSemiJoin', 'AntiLeftOuterSemiJoin')


def extract_join_type(operator):
    # operator.replace(' ', '')
    index = operator.find(',')
    type_str = operator[: index]
    return type_str


def extract_join_conditions(operator):
    operator.replace(' ', '')
    conditions = []
    # find all conditions
    conditions_str = re.findall(r'(\w+\(.*?\))', operator)
    for condition in conditions_str:
        function = re.search(r'(\w+)\(', condition).group(1)
        args = re.search(r'\((.*?)\)', condition).group(1).split(',')
        conditions.append(Condition(function, args))
    return conditions


def extract_selection_info(operator):
    conditions = []
    remain_operator = operator
    # iterator to extract conditions
    while len(remain_operator) > 0:
        match_one = re.search(r'^(.*?),\s+\w+\(', remain_operator)
        if match_one == None:
            condition = remain_operator  # remain treat as one condition
            remain_operator = ''
        else:
            condition = match_one.group(1)
            remain_operator = remain_operator[len(condition):]
            remain_operator = re.sub(r'^[,]\s+', '', remain_operator)

        condition_match = re.search(r'(\w+)\((.*?)\)$', condition)
        assert condition_match != None
        function, args = condition_match.group(
            1), condition_match.group(2)
        args = args.replace(' ', '').split(',')
        conditions.append(Condition(function, args))
    return conditions


def extract_table_reader(node):
    childrens = node['children']
    table_reader = TableReader()
    # childrens == 1?
    assert len(childrens) > 0
    q = queue.Queue()
    q.put(node)
    # bfs all child to find table and operatorInfo
    while not q.empty():
        cur_node = q.get()
        if 'Selection' in cur_node['id']:
            # extract operator info
            operator_info = cur_node['operatorInfo']
            conditions = extract_selection_info(operator_info)
            table_reader.conditions = conditions
        if 'Scan' in cur_node['id']:
            # extract which table to read
            assert 'accessObject' in cur_node
            access_object = cur_node['accessObject']
            table_name = access_object.split(':')[1]
            table_reader.table = table_name
        if 'children' in cur_node and cur_node['children'] != None:
            for item in cur_node['children']:
                q.put(item)

        table_reader.execute_time = node["AnalyzeInfo"]["time"] if "time" in node["AnalyzeInfo"] else "0s"
    return table_reader


def extract_join_info(node):
    r"""
        Recursive extract join tree from the root node data
        data node must be join type.
        ref: https://docs.pingcap.com/tidb/dev/explain-overview
    """
    operator_info = node['operatorInfo']
    analyze_info = node['AnalyzeInfo']

    if 'Join' in node['id']:
        # Join Node
        join_type = extract_join_type(operator_info)
        conditions = extract_join_conditions(operator_info)
        current_node = JoinPlan(join_type, conditions)
        assert 'children' in node and len(node['children']) == 2
        childrens = node['children']
        current_node.left_node = extract_join_info(childrens[0])
        current_node.right_node = extract_join_info(childrens[1])
        current_node.execute_time = analyze_info["time"]
    else:
        # Table Reader
        # assert 'TableReader' in node['id']
        # extract selection if need
        current_node = extract_table_reader(node)
    return current_node


def extract_join_tree(data_path):
    data = None
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # iterate data find the root join node
    current_node = data
    # TODO check join type
    while current_node:
        current_type = current_node['id']
        if('Join' in current_type):
            return extract_join_info(current_node)
        elif 'children' in current_node and current_node['children'] is not None:
            # just iterate the first children
            current_node = current_node['children'][0]
        else:
            current_node = None
    return current_node


class State():
    r""" An state corresponed to s_t of all trajectories,
    includes `current_join_tree` and `not join tables`"""

    def __init__(self, joined_tables, join_tree, not_join_tables):
        self.joined_tables = joined_tables
        self.not_join_tables = not_join_tables
        self.join_tree = join_tree
        self.is_done = len(not_join_tables) == 0

    def __str__(self):
        return f'joined_tables:{self.joined_tables},join_tree:{self.join_tree}not_join_tables:{self.not_join_tables}'

    def encode(self):
        return self.join_tree.encode()


class Action():
    r""" An action inclues join_table (inner), and join_conditions """

    def __init__(self, join_table, conditions):
        self.join_table = join_table
        self.conditions = conditions

    def __str__(self):
        return f"action:{self.join_table}"

    def encode(self):
        return self.join_table.encode()


class Observation():
    r""" An observation at time T includes [state, action,reward]
        action means which table be joined with join_tree at state
        reward is the actual execution time
    """

    def __init__(self, state, action=None, reward=0):
        self.state = state
        self.action = action
        self.reward = reward

    def __str__(self):
        return f'state:{self.state},action:{self.action},reawrd:{self.reward}'

    def __repr__(self):
        return self.__str__()


def convert_tree_to_trajectory(node):
    r"""Convert an join tree to trajectory.

    An trajectory includes[state, reward, next state]
    state includes current `join tree` and `not join tables`

    Args:
        node: the root of left deep join tree,
    """
    # TODO support bushy tree
    current_node = node
    # Deep first search tree, and extract state and reward
    # action at is None means terminal, so reawrd is zero
    join_trees = [current_node]  # final state
    join_order_reverse = []

    rewards = [0]
    # From end of the trajectory(st) to search(the root of the join tree)
    while type(current_node) == JoinPlan:
        action = Action(current_node.right_node, current_node.conditions)
        join_order_reverse.append(action)
        join_trees.append(current_node)
        rewards.append(current_node.execute_time)
        current_node = current_node.left_node
    else:
        join_order_reverse.append(
            Action(current_node, current_node.conditions))
        join_trees.append(current_node)
        # End of the search current_node is the first join table
    assert type(current_node) == TableReader
    assert len(rewards) == len(join_order_reverse)
    assert len(join_trees) == len(join_order_reverse)+1
    # First join table is current_node
    # Reverse actions and rewards ordered by time
    join_order_reverse.reverse()
    rewards.reverse()
    join_trees.reverse()
    actions_length = len(join_order_reverse)
    # Iterate actions and foramt trajectories
    trajectories = []
    print(join_order_reverse)
    print(rewards)
    for i in range(1, actions_length):
        state = State(join_order_reverse[:i],
                      join_trees[i-1], join_order_reverse[i:])
        trajectories.append(
            Observation(state, join_order_reverse[i], rewards[i-1]))
    else:
        state = State(join_order_reverse, join_trees[-1], [])
        trajectories.append(Observation(state, None, 0))
    # Construct trajectories
    print([t.state.is_done for t in trajectories])
    return trajectories
