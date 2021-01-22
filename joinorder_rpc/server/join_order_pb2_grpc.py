# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from . import join_order_pb2 as join__order__pb2


class GreeterStub(object):
    """The greeting service definition.
    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.SayHello = channel.unary_unary(
            '/join_order.Greeter/SayHello',
            request_serializer=join__order__pb2.HelloRequest.SerializeToString,
            response_deserializer=join__order__pb2.HelloReply.FromString,
        )
        self.SayHelloAgain = channel.unary_unary(
            '/join_order.Greeter/SayHelloAgain',
            request_serializer=join__order__pb2.HelloRequest.SerializeToString,
            response_deserializer=join__order__pb2.HelloReply.FromString,
        )


class GreeterServicer(object):
    """The greeting service definition.
    """

    def SayHello(self, request, context):
        """Sends a greeting
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def SayHelloAgain(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_GreeterServicer_to_server(servicer, server):
    rpc_method_handlers = {
        'SayHello': grpc.unary_unary_rpc_method_handler(
            servicer.SayHello,
            request_deserializer=join__order__pb2.HelloRequest.FromString,
            response_serializer=join__order__pb2.HelloReply.SerializeToString,
        ),
        'SayHelloAgain': grpc.unary_unary_rpc_method_handler(
            servicer.SayHelloAgain,
            request_deserializer=join__order__pb2.HelloRequest.FromString,
            response_serializer=join__order__pb2.HelloReply.SerializeToString,
        ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
        'join_order.Greeter', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))

 # This class is part of an EXPERIMENTAL API.


class Greeter(object):
    """The greeting service definition.
    """

    @staticmethod
    def SayHello(request,
                 target,
                 options=(),
                 channel_credentials=None,
                 call_credentials=None,
                 insecure=False,
                 compression=None,
                 wait_for_ready=None,
                 timeout=None,
                 metadata=None):
        return grpc.experimental.unary_unary(request, target, '/join_order.Greeter/SayHello',
                                             join__order__pb2.HelloRequest.SerializeToString,
                                             join__order__pb2.HelloReply.FromString,
                                             options, channel_credentials,
                                             insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def SayHelloAgain(request,
                      target,
                      options=(),
                      channel_credentials=None,
                      call_credentials=None,
                      insecure=False,
                      compression=None,
                      wait_for_ready=None,
                      timeout=None,
                      metadata=None):
        return grpc.experimental.unary_unary(request, target, '/join_order.Greeter/SayHelloAgain',
                                             join__order__pb2.HelloRequest.SerializeToString,
                                             join__order__pb2.HelloReply.FromString,
                                             options, channel_credentials,
                                             insecure, call_credentials, compression, wait_for_ready, timeout, metadata)


class JoinOrderStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.TestJoinNode = channel.unary_unary(
            '/join_order.JoinOrder/TestJoinNode',
            request_serializer=join__order__pb2.LogicalJoinNode.SerializeToString,
            response_deserializer=join__order__pb2.HelloReply.FromString,
        )
        self.GetAction = channel.unary_unary(
            '/join_order.JoinOrder/GetAction',
            request_serializer=join__order__pb2.State.SerializeToString,
            response_deserializer=join__order__pb2.HelloReply.FromString,
        )


class JoinOrderServicer(object):
    """Missing associated documentation comment in .proto file."""

    def TestJoinNode(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetAction(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_JoinOrderServicer_to_server(servicer, server):
    rpc_method_handlers = {
        'TestJoinNode': grpc.unary_unary_rpc_method_handler(
            servicer.TestJoinNode,
            request_deserializer=join__order__pb2.LogicalJoinNode.FromString,
            response_serializer=join__order__pb2.HelloReply.SerializeToString,
        ),
        'GetAction': grpc.unary_unary_rpc_method_handler(
            servicer.GetAction,
            request_deserializer=join__order__pb2.State.FromString,
            response_serializer=join__order__pb2.HelloReply.SerializeToString,
        ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
        'join_order.JoinOrder', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))

 # This class is part of an EXPERIMENTAL API.


class JoinOrder(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def TestJoinNode(request,
                     target,
                     options=(),
                     channel_credentials=None,
                     call_credentials=None,
                     insecure=False,
                     compression=None,
                     wait_for_ready=None,
                     timeout=None,
                     metadata=None):
        return grpc.experimental.unary_unary(request, target, '/join_order.JoinOrder/TestJoinNode',
                                             join__order__pb2.LogicalJoinNode.SerializeToString,
                                             join__order__pb2.HelloReply.FromString,
                                             options, channel_credentials,
                                             insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetAction(request,
                  target,
                  options=(),
                  channel_credentials=None,
                  call_credentials=None,
                  insecure=False,
                  compression=None,
                  wait_for_ready=None,
                  timeout=None,
                  metadata=None):
        return grpc.experimental.unary_unary(request, target, '/join_order.JoinOrder/GetAction',
                                             join__order__pb2.State.SerializeToString,
                                             join__order__pb2.HelloReply.FromString,
                                             options, channel_credentials,
                                             insecure, call_credentials, compression, wait_for_ready, timeout, metadata)