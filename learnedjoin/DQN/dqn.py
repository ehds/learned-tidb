import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from learnedjoin.DQN.replay_buffer import ReplayBuffer
import learnedjoin.DQN.tcnn as tcnn
from utils.join_order import extract_join_tree, convert_tree_to_trajectory
from learnedjoin.DQN.tree_lstm import TreeTLSTMQFunction
from datetime import datetime
import os
import math
import random


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
    def __init__(self, database, workload, buffer_size, batch_size, obs_dim, act_dim, latency_train=False):
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
        # self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = 'cpu'
        self.replay_buffer = ReplayBuffer(buffer_size, batch_size)
        # self.q_net = MLPQFunction(obs_dim, act_dim, [32])
        # self.q_net = TreeCNNQFunction(obs_dim, 32, act_dim)
        self.q_net = TreeTLSTMQFunction(
            obs_dim, database.get_all_tables(), database.unique_columns, self.device)
        self.target_net = TreeTLSTMQFunction(
            obs_dim, database.get_all_tables(), database.unique_columns, self.device)
        self.target_net.eval()
        # self.tree_lstm = TreeLSTM(obs_dim)
        self.train_steps = 2000
        self.batch_size = batch_size
        self.gamma = 0.99
        self.clippng_norm = 10
        self.steps_done = 0
        self.q_net.to(self.device)
        self.latency_train = latency_train
        self.load_model()

    def get_best_action(self, state, actions):

        # get encoding of state and actions
        eps = 0.05+(0.9)*math.exp(-1. * self.steps_done / 200)
        self.steps_done = self.steps_done+1
        if random.random() > eps:
            costs = []
            with torch.no_grad():
                for action in actions:
                    # q_value batch*1
                    q_value = self.q_net([state], [action])
                costs.append(q_value.flatten().cpu().numpy()[0])
            return np.argmin(costs)
        else:
            return random.randrange(len(actions))

    def act(self, num=1):
        r""" Act num query to the database for collecting some actual 
        physical plan and execution time"""
        assert num > 0
        queries = self.workload.sample(num)
        # queries = self.workload.get_all_query()
        # TODO parallel execute
        for query in queries:
            # get actual  execution info
            execution_info = self.database.explain(query, self.latency_train)
            if execution_info == None:
                continue
            # add to replay buffer TODO save it
            join_tree = extract_join_tree(execution_info)
            trajectories = convert_tree_to_trajectory(
                join_tree, self.latency_train)
            for i in range(len(trajectories)-1):
                ob, next_ob = trajectories[i], trajectories[i+1]
                self.replay_buffer.add_observation(
                    ob.state, ob.action, ob.reward, next_ob.state, next_ob.state.is_done)

    def save_model(self):
        # date_str = datetime.now().strftime('%m-%d-%Y-%H-%M-%S')
        path = os.path.join('model', 'dqn.pth')
        torch.save(self.q_net.state_dict(), path)

    def load_model(self):
        path = os.path.join('model', 'dqn.pth')
        if os.path.exists(path):
            self.q_net.load_state_dict(torch.load(path))
            self.q_net.train()

    def train(self):
        # TODO off-line to on-line

        q_optimizer = Adam(self.q_net.parameters(), lr=1e-3)
        best_mrc = 1e10
        self.q_net.train()
        losses = []
        for step in range(self.train_steps):

            # get batch data
            # act
            self.act(num=4)
            batch = self.replay_buffer.sample(self.batch_size, False)
            states, actions, rewards, next_states, dones = batch
            rewards = torch.FloatTensor(rewards).to(self.device)
            dones = torch.FloatTensor(dones).to(self.device)

            Q = self.q_net(states, actions)
            # compute expeacted Q(s_i+1,pi(a))
            # TODO search a

            Q_next = self.target_net(next_states, actions)

            Q_expected = rewards+self.gamma*Q_next*(1-dones)

            loss = F.smooth_l1_loss(Q, Q_expected, reduction='mean')
            losses.append(loss.item())
            # update network
            q_optimizer.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm_(
            #     self.q_net.parameters(), self.clippng_norm)
            q_optimizer.step()
            if (step+1) % 50 == 0:
                print(len(losses))
                print("mean loss:", np.mean(losses))
                mean, mrc = self.validate()
                print(f"mean:{mean}, mrc:{mrc}")
                if mrc > best_mrc:
                    best_mrc = mrc
                self.save_model()
                self.target_net.load_state_dict(self.q_net.state_dict())

    def set_dqn(self, flag):
        with open('config.conf', 'w') as f:
            f.write(str(flag))

    def validate(self):
        validate_set = self.workload.get_all_query()
        # change server policy to greedy
        self.q_net.eval()

        perfomance_ratio = []
        for q in validate_set[:10]:
            # get db original cost
            self.set_dqn(False)
            db_cost = self.database.get_latency(q)
            self.set_dqn(True)
            dqn_cost = self.database.get_latency(q)
            perfomance_ratio.append(math.log(dqn_cost)-math.log(db_cost))

        mean = np.mean(np.exp(perfomance_ratio))
        mrc = np.exp(np.sum(perfomance_ratio)/len(perfomance_ratio))
        return mean, mrc
