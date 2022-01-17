import numpy as np
import torch
from torch import nn
from torch.distributions import Normal


class ActorCritic(nn.Module):

    def __init__(self, state_size, action_size, hidden_layer=None):
        super(ActorCritic, self).__init__()
        if hidden_layer is None: hidden_layer = [64, 64]
        hidden_layer.insert(0, state_size)

        base_layer = []
        for i in range(len(hidden_layer) - 1):
            base_layer.append(nn.Linear(hidden_layer[i], hidden_layer[i + 1]))
        self.base = nn.Sequential(*base_layer)

        self.actor = nn.Linear(hidden_layer[-1], action_size)  # output mean
        self.critic = nn.Linear(hidden_layer[-1], 1)

    def forward(self):
        pass

    def act(self, state):
        if isinstance(state, np.ndarray):
            state = torch.tensor(state).float()
        state_feature = self.base(state)
        act_mean = self.actor(state_feature)
        act_std = torch.ones_like(act_mean)

        dist = Normal(act_mean, act_std)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.detach().cpu().numpy(), log_prob

    def evaluate(self, state):
        if isinstance(state, np.ndarray):
            state = torch.tensor(state).float()
        state_feature = self.base(state)
        action = self.critic(state_feature)
        return action
