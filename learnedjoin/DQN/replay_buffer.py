from collections import namedtuple, deque
import random
import torch
import numpy as np


class ReplayBuffer(object):
    def __init__(self, buffer_size, batch_size, device=None):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.observation = namedtuple("Observation", field_names=[
            "state", "action", "reward", "next_state", "done"])
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")

    def add_observation(self, states, actions, rewards, next_states, dones):
        if type(dones) == list:
            assert type(dones[0]) != list, "A done shouldn't be a list"
            experiences = [self.observation(state, action, reward, next_state, done)
                           for state, action, reward, next_state, done in
                           zip(states, actions, rewards, next_states, dones)]
            self.memory.extend(experiences)
        else:
            experience = self.observation(
                states, actions, rewards, next_states, dones)
            self.memory.append(experience)

    def sample(self, num_experiences=None, separate_out_data_types=True):
        batch_size = self.batch_size if num_experiences is None else num_experiences
        batch_size = min(batch_size, len(self.memory))
        observations = random.sample(self.memory, k=batch_size)
        if separate_out_data_types:
            states, actions, rewards, next_states, dones = self.separate_out_data_types(
                observations)
            return states, actions, rewards, next_states, dones
        else:
            return ([o.state for o in observations], [o.action for o in observations],
                    [o.reward for o in observations], [
                o.next_state for o in observations],
                [o.done for o in observations])

    def separate_out_data_types(self, experiences):
        """Puts the sampled experience into the correct format for a PyTorch neural network"""
        states = torch.from_numpy(np.stack(
            [e.state[0] for e in experiences if e is not None], axis=0)).float().to(self.device)
        tree_conv = torch.from_numpy(np.stack(
            [e.state[1] for e in experiences if e is not None], axis=0)).long().to(self.device)
        actions = torch.from_numpy(np.stack(
            [e.action[0] for e in experiences if e is not None], axis=0)).float().to(self.device)
        tree_action = torch.from_numpy(np.stack(
            [e.action[1] for e in experiences if e is not None], axis=0)).long().to(self.device)
        rewards = torch.from_numpy(np.vstack(
            [e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.stack(
            [e.next_state[0] for e in experiences if e is not None], axis=0)).float().to(self.device)
        next_tree_conv = torch.from_numpy(np.stack(
            [e.state[1] for e in experiences if e is not None], axis=0)).long().to(self.device)
        dones = torch.from_numpy(np.vstack(
            [int(e.done) for e in experiences if e is not None])).float().to(self.device)
        return (states, tree_conv), (actions, tree_action), rewards, (next_states, next_tree_conv), dones

    def __len__(self):
        return len(self.memory)
