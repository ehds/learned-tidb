import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from replay_buffer import ReplayBuffer


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


class MLPQFunction(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation=nn.ReLU):
        super(MLPQFunction, self).__init__()
        self.q = mlp(obs_dim+act_dim+list(hidden_sizes)+[1], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1)


class TreeCNNQFunction(nn.Module):
    pass


class TreeLSTMQFuntion(nn.Module):
    pass


class JoinOrderDQN():
    def __init__(self, buffer_size, batch_size, obs_dim, act_dim):
        self.replay_buffer = ReplayBuffer(buffer_size, batch_size)
        self.q_net = MLPQFunction(obs_dim, act_dim, [125, 64, 32])
        self.train_steps = 100
        self.batch_size = batch_size
        self.gamma = 0.99
        self.clippng_norm = 10

    def train(self):
        # TODO off-line to on-line

        q_optimizer = Adam(self.q_net.parameters(), lr=1e-3)
        self.q_net.train()
        for step in range(self.train_steps):
            # get batch data
            batch = self.replay_buffer.sample(self.batch_size)
            states, actions, rewards, next_states, dones = batch
            Q = self.q_net(states, actions)
            # compute expeacted Q(s_i+1,pi(a))
            # TODO search a
            with torch.no_grad():
                Q_next = self.q_net(next_states, actions)
            Q_expected = rewards+self.gamma*Q_next*(1-dones)
            loss = F.mse_loss(Q_expected, Q_next)
            # update network
            q_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                self.q_net.parameters(), self.clippng_norm)
            q_optimizer.step()
