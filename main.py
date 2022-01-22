import math

import gym
import numpy as np
import torch
from matplotlib import pyplot as plt
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv
import torch.nn.functional as F
from torch import optim, nn

from ActorCritic import ActorCritic
from Buffer import Buffer


def make_env(env_id, rank, seed=0):
    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)
        return env

    set_random_seed(seed)
    return _init


total_steps = 1e3
num_process = 3  # num of env running in parallel
learn_interval = 1e2  # agent learn per 1e3 steps
# Note that the buffer has to collect at least `learn_interval` experiences
# consider `num_process`, envs should be interacted for `learn_interval/num_process` steps
# which is also the capacity of buffer
batch_size = 10
learn_epoch = 10  # ppo learns 10 times each time learn

rewards_to_plot = []
if __name__ == '__main__':
    # set_random_seed(0)  # make the program reproducible
    # set up envs
    envs = SubprocVecEnv([make_env("Pendulum-v0", i) for i in range(num_process)])

    # set up actor critic
    state_size = envs.observation_space.shape[0]

    action_space = envs.action_space
    action_size = None
    if action_space.__class__.__name__ == "Discrete":
        action_size = action_space.n
    elif action_space.__class__.__name__ == "Box":
        action_size = action_space.shape[0]
    else:
        print(f'unknown action space')
    ac = ActorCritic(state_size, action_size)
    print(ac)
    optimizer = optim.Adam(ac.parameters(), lr=learning_rate)

    # set up buffer
    buffer = Buffer(int(learn_interval // num_process), int(batch_size))

    learn_times = math.ceil(total_steps / learn_interval)
    for n in range(learn_times):
        # initial buffer and envs
        buffer.reset()
        state = envs.reset()  # todo: how to handle next state
        buffer.states[0] = state  # initial state
        episode_reward = [0] * num_process
        episode_rewards = []

        # interact with envs
        envs_steps = math.ceil(learn_interval / num_process)
        # each time learn, envs should run for these steps to fill the buffer, i.e. prepare the data
        for _ in range(envs_steps):
            with torch.no_grad():
                action, action_log_prob = ac.act(state)
                state_value = ac.evaluate(state)
            state, rewards, dones, infos = envs.step(action)
            episode_reward = [a + b for a, b in zip(episode_reward, rewards)]
            for i, done in enumerate(dones):
                if done:
                    episode_rewards.append(episode_reward[i])
                    episode_reward[i] = 0

            buffer.add(state, state_value.numpy(), action, action_log_prob.numpy(), rewards, dones)
            # NOTE that we are storing `next_state` and `current_state_value`

        with torch.no_grad():
            state_value = ac.evaluate(state)
        buffer.state_values[-1] = state_value.detach().numpy()  # last state value
        assert buffer.capacity == buffer.size, 'the buffer is not full'

        buffer.compute_gae()
        buffer.compute_discount_rewards()

        # start to learn
        actor_loss_list = []
        critic_loss_list = []
        for i in range(learn_epoch):
            for states, state_values, actions, old_action_log_probs, advantages, discount_rewards, returns in buffer.data_generator():
                # actor loss
                action_log_probs = ac.get_action_log_prob(states, actions)
                ratio = torch.exp(action_log_probs - old_action_log_probs)
                ratio_clipped = torch.clip(ratio, 1 - clip_ratio, 1 + clip_ratio)
                actor_loss = -torch.min(ratio * advantages, ratio_clipped * advantages).mean()
                assert actor_loss.requires_grad
                actor_loss_list.append(float(actor_loss.data))

                # critic loss
                new_state_values = ac.evaluate(states)  # todo: seems the state values from batch is not used
                # critic_loss = F.mse_loss(new_state_values, returns, reduction='mean')
                # critic_loss = F.mse_loss(new_state_values, advantages + state_values, reduction='mean')
                critic_loss = F.mse_loss(new_state_values, discount_rewards, reduction='mean')
                assert critic_loss.requires_grad
                critic_loss_list.append(float(critic_loss.data))

                # update actor critic
                loss = actor_loss + critic_loss * 0.5
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(ac.parameters(), 0.5)  # todo: clip grad
                optimizer.step()

        buffer.reset()
        print(f'{"-" * 10} update {n + 1}, experience collected: {buffer.size * num_process} {"-" * 10}\n'
              f'{len(episode_rewards)} episodes finished, '
              f'mean: {np.mean(episode_rewards): .4f}, std: {np.std(episode_rewards): .4f},'
              f'actor loss: {np.mean(actor_loss_list): .4f}, critic loss: {np.mean(critic_loss_list): .4f}'
              )
        rewards_to_plot.append(np.mean(episode_rewards))

    # plot episode reward
    fig, ax = plt.subplots()
    ax.plot(range(1, len(rewards_to_plot) + 1), rewards_to_plot)
    ax.hlines(-165, 1, len(rewards_to_plot) + 1, colors='red')
    plt.savefig(f'Pendulum-v0 result.png')


