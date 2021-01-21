from concurrent import futures


import grpc

from . import join_order_pb2
from . import join_order_pb2_grpc


class Greeter(join_order_pb2_grpc.GreeterServicer):

    def SayHello(self, request, context):
        print(request.name)
        return join_order_pb2.HelloReply(message='Hello, %s!' % request.name)


class JoinOrder(join_order_pb2_grpc.JoinOrderServicer):
    def TestJoinNode(self, logical_node, context):
        print(logical_node.tp)
        for item in logical_node.conditions:
            print(item.funcname, item.args)
        return join_order_pb2.HelloReply(message='Hello')

    def GetAction(self, state, context):
        # print(state.current_join_tree.tp)
        print(len(state.actions))
        for item in state.actions:
            print(item.tp)
            assert(item.tp == 'DataSource')
        return join_order_pb2.HelloReply(message='hello')


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    join_order_pb2_grpc.add_GreeterServicer_to_server(Greeter(), server)
    join_order_pb2_grpc.add_JoinOrderServicer_to_server(JoinOrder(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()
