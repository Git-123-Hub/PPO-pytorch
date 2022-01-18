import numpy as np
import torch
from torch.utils.data import BatchSampler, SubsetRandomSampler


class Buffer:
    def __init__(self, capacity, batch_size, gamma=0.99, gae_lambda=0.95):
        self.capacity = capacity
        self._index = 0
        self._size = 0
        self.batch_size = batch_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.states = np.zeros(capacity + 1, dtype=object)  # store initial state
        self.actions = np.zeros(capacity, dtype=object)
        self.rewards = np.zeros(capacity, dtype=object)
        self.dones = np.zeros(capacity + 1, dtype=object)

        self.state_values = np.zeros(capacity + 1, dtype=object)
        self.action_log_probs = np.zeros(capacity, dtype=object)

        self.advantages = np.zeros(capacity, dtype=object)

    def reset(self):
        self._index = 0
        self.states.fill(0)
        self.actions.fill(0)
        self.rewards.fill(0)
        self.dones.fill(0)
        self.state_values.fill(0)
        self.action_log_probs.fill(0)
        self.advantages.fill(0)

    def add(self, state, state_value, action, action_log_prob, reward, done):
        self.states[self._index + 1] = state
        # [self._index] is current state, [self._index + 1] is next state
        self.state_values[self._index] = state_value
        # [self._index] is current state value, [self._index + 1] is next state value

        self.actions[self._index] = action
        self.action_log_probs[self._index] = action_log_prob
        self.rewards[self._index] = reward
        self.dones[self._index] = done

        self._index = (self._index + 1) % self.capacity
        if self._size < self.capacity:
            self._size += 1

    def compute_gae(self):
        gae = 0
        for i in reversed(range(self._size)):
            delta = self.rewards[i] \
                    + self.state_values[i + 1] * (1 - self.dones[i + 1]) * self.gamma \
                    - self.state_values[i]
            gae = delta + self.gamma * self.gae_lambda * (1 - self.dones[i + 1]) * gae
            self.advantages[i] = gae

    def data_generator(self):
        # reshape all the data
        all_states = np.stack(self.states).reshape(-1, self.states[0].shape[-1])
        all_state_values = np.stack(self.state_values).reshape(-1)
        all_actions = np.stack(self.actions).reshape(-1, self.actions[0].shape[-1])
        all_log_probs = np.stack(self.action_log_probs).reshape(-1)
        all_advantages = np.stack(self.advantages).reshape(-1)

        total_num = self._size * self.states[0].shape[0]  # size * num_process
        index_sampler = BatchSampler(SubsetRandomSampler(range(total_num)), self.batch_size, drop_last=True)
        for indices in index_sampler:
            # retrieve data and transfer to tensor
            states = torch.from_numpy(all_states[indices]).float()
            state_values = torch.from_numpy(all_state_values[indices]).float()
            actions = torch.from_numpy(all_actions[indices]).float()
            action_log_probs = torch.from_numpy(all_log_probs[indices]).float()
            advantages = torch.from_numpy(all_advantages[indices]).float()
            yield states, state_values, actions, action_log_probs, advantages
