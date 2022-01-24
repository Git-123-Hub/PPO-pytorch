import numpy as np
import torch
from torch import nn
from torch.distributions import Normal


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class ActorCritic(nn.Module):

    def __init__(self, state_size, action_size, hidden_layer=None):
        super(ActorCritic, self).__init__()
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))
        init_1 = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))
        self.actor = nn.Sequential(
            init_(nn.Linear(state_size, 64)),
            nn.Tanh(),
            init_(nn.Linear(64, 64)),
            nn.Tanh(),
            init_1(nn.Linear(64, action_size)),
        )

        self.critic = nn.Sequential(
            init_(nn.Linear(state_size, 64)),
            nn.Tanh(),
            init_(nn.Linear(64, 64)),
            nn.Tanh(),
            init_(nn.Linear(64, 1))
        )

        self.std = 1

    def forward(self):
        raise NotImplementedError

    def get_dist(self, state):
        """get the action distribution under `state`"""
        if isinstance(state, np.ndarray):
            state = torch.tensor(state).float()
        act_mean = self.actor(state)
        act_std = torch.ones_like(act_mean) * self.std
        dist = Normal(act_mean, act_std)
        return dist

    def act(self, state):
        dist = self.get_dist(state)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=1)
        return action.detach().cpu().numpy(), log_prob

    def get_action_log_prob(self, state, action):
        """get the log prob of execute action in state"""
        dist = self.get_dist(state)
        log_prob = dist.log_prob(action).sum(dim=1)
        return log_prob

    def evaluate(self, state):
        """get the estimated state value of `state`"""
        if isinstance(state, np.ndarray):
            state = torch.tensor(state).float()
        state_value = self.critic(state).squeeze(dim=1)
        return state_value
