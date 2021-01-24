from . import join_order_pb2
from . import join_order_pb2_grpc
from utils.join_order import JoinPlan, TableReader, Condition


def encode_logical_plan(node):
    current_node = None
    node_type = node.WhichOneof('node')
    if node_type == 'join_node':
        join_node = node.join_node
        current_node = JoinPlan(join_node.join_type)
        current_node.conditions = [Condition(item.funcname, item.args)
                                   for item in join_node.conditions]
        assert len(node.childrens) == 2
        current_node.left_node = encode_logical_plan(node.childrens[0])
        current_node.right_node = encode_logical_plan(node.childrens[1])

    elif node_type == 'table_node':
        table_node = node.table_node
        current_node = TableReader(table_node.table_name)
        current_node.conditions = [
            Condition(item.funcname, item.args) for item in table_node.conditions]
    return current_node
