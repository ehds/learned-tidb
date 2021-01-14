import json
from enum import Enum
import re


class Plan():
    def __init__(self):
        self.children = []
        self.id = 0
        # -1 means nothing
        self.execute_time = -1
        self.cost = -1

    def encoding(self):
        raise NotImplementedError()


class JoinPlan(Plan):
    def __init__(self, join_type, conditions):
        super(JoinPlan, self).__init__()
        self.join_type = join_type
        self.conditions = conditions
        self.left_node = None
        self.right_node = None

    def encode(self):
        raise NotImplementedError()

    def __str__(self):
        return f"{self.join_type},{self.conditions},left:{self.left_node},right:{self.right_node}"

    def __repr__(self):
        return self.__str__()


class TableReader(Plan):
    def __init__(self, table='', conditions=[]):
        self.table = table
        self.conditions = conditions

    def encode(self):
        raise NotImplementedError()

    def __str__(self):
        return f"reader: {self.table},{self.conditions}"

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
    #operator.replace(' ', '')
    index = operator.find(',')
    type_str = operator[:index]
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


def extract_table_reader(operator):
    children = operator['children']
    table_reader = TableReader()
    assert len(children) > 0
    for child in children:
        if 'Selection' in child['id']:
            # extract operator info
            operator_info = child['operatorInfo']
            conditions = extract_selection_info(operator_info)
            table_reader.conditions = conditions
        if 'Scan' in child['id']:
            # extract which table to read
            assert 'accessObject' in child
            access_object = child['accessObject']
            table_name = access_object.split(':')[1]
            table_reader.table = table_name
    return table_reader


def extract_join_info(node):
    r"""
        Recursive extract join tree from the root node data
        data node must be join type
    """
    operator_info = node['operatorInfo']
    curren_node = None

    if 'Join' in node['id']:
        # Join Node
        join_type = extract_join_type(operator_info)
        conditions = extract_join_conditions(operator_info)
        current_node = JoinPlan(join_type, conditions)
        assert 'children' in node and len(node['children']) == 2
        childrens = node['children']
        current_node.left_node = extract_join_info(childrens[0])
        current_node.right_node = extract_join_info(childrens[1])
    else:
        # Table Reader
        assert 'TableReader' in node['id']
        # extract selection if need
        current_node = extract_table_reader(node)
    return current_node


def extract_join_tree(data_path):
    data = None
    with open(data_path, 'r') as f:
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


def convert_tree_to_trajectory(node):
    r"""Convert an join tree to trajectory.

    An trajectory includes[state,reward,next state]
    state includes current `join tree` and `not join tables`

    Args:
        node: the root of left deep join tree, 
    """
    # deep first search tree, and extract state and reward
    trajectories = []
