import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from learnedjoin.DQN.replay_buffer import ReplayBuffer
import learnedjoin.DQN.tcnn as tcnn
from utils.join_order import extract_join_tree, convert_tree_to_trajectory


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


class MLPQFunction(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation=nn.ReLU):
        super(MLPQFunction, self).__init__()
        self.q = mlp([obs_dim+act_dim]+list(hidden_sizes)+[1], activation)

    def forward(self, obs, act):
        input = torch.cat((obs, act), dim=-1)
        output = self.q(input)
        return output


class TreeCNNQFunction(nn.Module):
    def __init__(self, obs_dim, hidden_dim, act_dim):
        r""" An tree CNN Q function
        args:
            obs_dim: the node dim of obs tree
            hidden_dim: encodings dim of obs tree
            act_dim: action dim
        """
        # TODO hiddent_size need to be set
        super(TreeCNNQFunction, self).__init__()
        self.state_encoder = StateTreeEncoder(obs_dim, hidden_dim)
        self.action_encode = StateTreeEncoder(act_dim, hidden_dim)
        self.q = mlp([2*hidden_dim] +
                     [125, 32, 4]+[1], activation=nn.ReLU)

    def forward(self, states, actions):
        # batch * hidden_dim
        encodings = self.state_encoder(states)
        actions = self.action_encode(actions)
        inputs = torch.cat((encodings, actions), dim=1)
        return self.q(inputs)
        # batch * 1 rewards
        return inputs


class TreeLSTMQFuntion(nn.Module):
    pass


class StateTreeEncoder(nn.Module):
    def __init__(self, obs_dim, output_dim):
        super(StateTreeEncoder, self).__init__()
        self.obs_dim = obs_dim
        self.output_dim = output_dim
        self.tree_conv_net = nn.Sequential(
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

    def forward(self, inputs):
        return self.tree_conv_net(inputs)


class JoinOrderDQN(object):
    def __init__(self, database, workload, buffer_size, batch_size, obs_dim, act_dim):
        r""" Join order policy model
            Args:
                database: which database you want to train
                workload: query sql set
                buffer_size: how many states to keep in buffer
                batch_size: batch size for every update
                obs_dim : current join state dimension
                act_dim :  join action dimension
        """
        # database to train
        self.database = database
        self.workload = workload
        self.replay_buffer = ReplayBuffer(buffer_size, batch_size)
        # self.q_net = MLPQFunction(obs_dim, act_dim, [32])
        self.q_net = TreeCNNQFunction(obs_dim, 32, act_dim)
        self.train_steps = 1000
        self.batch_size = batch_size
        self.gamma = 0.99
        self.clippng_norm = 10
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.q_net.to(self.device)
    def get_best_action(self, state, actions):
        costs = []
        # get encoding of state and actions
        current_join_tree = torch.Tensor([state[0]]).to(self.device)
        tree_index = torch.LongTensor([state[1]]).to(self.device)
        for action in actions:
            action_tree = torch.Tensor([action[0]]).to(self.device)
            action_index = torch.LongTensor([action[1]]).to(self.device)
            with torch.no_grad():
                # q_value batch*1
                q_value = self.q_net((current_join_tree, tree_index),
                                     (action_tree, action_index))
            costs.append(q_value.flatten().cpu().numpy()[0])
        return np.argmin(costs)

    def act(self, num=1):
        r""" Act num query to the database for collecting some actual 
        physical plan and execution time"""
        assert num > 0
        queries = self.workload.sample(num)
        # TODO parallel execute
        for query in queries:
            # get actual  execution info
            execution_info = self.database.analyze(query)
            if execution_info == None:
                continue
            # add to replay buffer TODO save it
            join_tree = extract_join_tree(execution_info)
            trajectories = convert_tree_to_trajectory(join_tree)
            for i in range(len(trajectories)-1):
                ob, next_ob = trajectories[i], trajectories[i+1]
                self.replay_buffer.add_observation(
                    ob.state.encode(), ob.action.encode(), ob.reward, next_ob.state.encode(), next_ob.state.is_done)

    def train(self):
        # TODO off-line to on-line

        q_optimizer = Adam(self.q_net.parameters(), lr=1e-3)

        self.q_net.train()
        losses = []
        for step in range(1,self.train_steps):
            q_optimizer.zero_grad()
            # get batch data
            # act
            self.act(num=1)
            batch = self.replay_buffer.sample(self.batch_size)
            states, actions, rewards, next_states, dones = batch
            Q = self.q_net(states, actions)
            # compute expeacted Q(s_i+1,pi(a))
            # TODO search a
            Q_next = 0
            with torch.no_grad():
                Q_next = self.q_net(next_states, actions)

            Q_expected = rewards+self.gamma*Q_next*(1-dones)

            loss = F.mse_loss(Q_expected, Q)
            losses.append(loss.item())
            # update network
            loss.backward()
            nn.utils.clip_grad_norm_(
                self.q_net.parameters(), self.clippng_norm)
            q_optimizer.step()
            if step % 50 == 0:
                print("mean loss:",np.mean(losses))
                losses = []
