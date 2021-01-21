from utils.join_order import extract_join_tree
import learnedjoin.DQN.tcnn as tcnn
import torch
import torch.nn as nn


def test_test_tree_cnn():
    a = extract_join_tree('tests/test.json')
    obs_dim = a.encode()[0].shape[-1]
    output_dim = 4
    net = nn.Sequential(
        tcnn.BinaryTreeConv(obs_dim, 16),
        tcnn.TreeLayerNorm(),
        tcnn.TreeActivation(nn.ReLU()),
        tcnn.BinaryTreeConv(16, 8),
        tcnn.TreeLayerNorm(),
        tcnn.TreeActivation(nn.ReLU()),
        tcnn.BinaryTreeConv(8, output_dim),
        tcnn.TreeLayerNorm(),
        tcnn.TreeActivation(nn.ReLU()),
        tcnn.DynamicPooling()
    )
    obs_states = torch.Tensor([a.encode()[0]])
    tree_conv_index = torch.LongTensor([a.encode()[1]])
    output = net((obs_states, tree_conv_index))
    assert output.shape[0] == 1
    assert output.shape[1] == output_dim
