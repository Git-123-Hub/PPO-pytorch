import math
import os
import pickle
import time
from collections import deque

import gym
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from torch import optim, nn

from ActorCritic import ActorCritic
from Buffer import Buffer
from arguments import get_args


def make_env(env_id, rank, seed=0):
    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)
        env = Monitor(env)
        return env

    return _init


rewards_to_plot = []
running_rewards = []
running_window = deque(maxlen=100)
if __name__ == '__main__':
    start_time = time.time()
    args = get_args()
    print(args)

    if args.seed is not None:
        set_random_seed(args.seed)  # make the program reproducible
        print(f'seed has been set to {args.seed}')

    print(f'each env interact for {args.learn_interval}, total num process: {args.num_process}, '
          f'learn for {args.total_steps / args.num_process / args.learn_interval: .2f} times, '
          f'learn epoch: {args.learn_epoch}, '
          f'with each time {args.learn_interval * args.num_process / args.batch_size: .2f} batches')

    # set up envs
    envs = SubprocVecEnv([make_env(args.env_name, i) for i in range(args.num_process)])
    envs = VecNormalize(envs, gamma=0.99)

    goal = gym.make(args.env_name).spec.reward_threshold
    if goal is None:
        goal = {'Pendulum-v0': -165, }[args.env_name]
    solved = False  # flag indicates whether the problem has been solved(running reward greater than goal)

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
    optimizer = optim.Adam(ac.parameters(), lr=args.learning_rate)

    # set up buffer
    # Note that the buffer has to collect at least `learn_interval` experiences
    # consider `num_process`, envs should be interacted for `learn_interval/num_process` steps
    # which is also the capacity of buffer
    buffer = Buffer(int(args.learn_interval), int(args.batch_size))

    learn_times = math.ceil(args.total_steps / args.learn_interval / args.num_process)
    for n in range(learn_times):
        # initial buffer and envs
        buffer.reset()
        state = envs.reset()  # todo: how to handle next state
        buffer.states[0] = state  # initial state

        # decay learning rate and action distribution std
        # todo: exponential
        if args.lr_decay:
            learning_rate = args.learning_rate - args.learning_rate * n / learn_times
            learning_rate = max(learning_rate, 5e-5)
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate

        if args.std_decay:
            std = 1 - 1 * n / learn_times
            std = max(std, 0.5)
            ac.std = std

        # interact with envs
        # each time learn, envs should run for these steps to fill the buffer, i.e. prepare the data
        for _ in range(args.learn_interval):
            with torch.no_grad():
                action, action_log_prob = ac.act(state)
                state_value = ac.evaluate(state)
            state, rewards, dones, infos = envs.step(action)
            # NOTE that `state` has been updated as the next state
            for i, done in enumerate(dones):
                if done:
                    rewards_to_plot.append(infos[i]['episode']['r'])
                    running_window.append(infos[i]['episode']['r'])
                    running_rewards.append(np.mean(running_window))
                    if not solved and running_rewards[-1] >= goal:
                        solved = True

            buffer.add(state, state_value.numpy(), action, action_log_prob.numpy(), rewards, dones)
            # NOTE that we are storing `next_state` and `current_state_value`

        with torch.no_grad():
            state_value = ac.evaluate(state)
        buffer.state_values[-1] = state_value.detach().numpy()  # last state value

        buffer.compute_gae()
        buffer.compute_discount_rewards()

        # start to learn
        actor_loss_list = []
        critic_loss_list = []
        for i in range(args.learn_epoch):
            for states, state_values, actions, old_action_log_probs, advantages, discount_rewards, returns in buffer.data_generator():
                # actor loss
                action_log_probs = ac.get_action_log_prob(states, actions)
                ratio = torch.exp(action_log_probs - old_action_log_probs)
                ratio_clipped = torch.clip(ratio, 1 - args.clip_ratio, 1 + args.clip_ratio)
                actor_loss = -torch.min(ratio * advantages, ratio_clipped * advantages).mean()
                assert actor_loss.requires_grad
                actor_loss_list.append(float(actor_loss.data))

                # critic loss
                new_state_values = ac.evaluate(states)  # todo: seems the state values from batch is not used
                critic_loss = F.mse_loss(new_state_values, returns, reduction='mean')
                # critic_loss = F.mse_loss(new_state_values, advantages + state_values, reduction='mean')
                # critic_loss = F.mse_loss(new_state_values, discount_rewards, reduction='mean')
                assert critic_loss.requires_grad
                critic_loss_list.append(float(critic_loss.data))

                # update actor critic
                loss = actor_loss + critic_loss
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(ac.parameters(), args.clip_grad)
                optimizer.step()

        print(
            f'{"-" * 10} update {n + 1}, running reward: {np.mean(running_window): .4f} {"-" * 10}\n'
            f'actor loss: {np.mean(actor_loss_list): .4f}, critic loss: {np.mean(critic_loss_list): .4f}, '
            f'learning rate: {learning_rate if "learning_rate" in locals() else args.learning_rate: .5f}, '
            f'action std: {std if "std" in locals() else 1: .4f}'
        )

    # folder to save fig and data
    directory = args.folder if args.folder is not None else args.env_name
    if not os.path.exists(directory):
        os.makedirs(directory)
        total_files = 0
    else:
        total_files = len([file for file in os.listdir(directory)]) // 2  # data and fig

    # training finishes, plot episode reward
    fig, ax = plt.subplots()
    ax.plot(range(1, len(rewards_to_plot) + 1), rewards_to_plot)
    ax.plot(range(1, len(running_rewards) + 1), running_rewards)
    ax.hlines(y=goal, xmin=1, xmax=len(rewards_to_plot) + 1, color='red')
    filename = f'{args.env_name} result {total_files + 1}'
    plt.savefig(os.path.join(directory, filename + '.png'))
    # save training data
    with open(os.path.join(directory, filename + '.pkl'), 'wb') as f:
        pickle.dump({
            'args': args,
            'rewards_to_plot': rewards_to_plot,
            'running_rewards': running_rewards,
            'solved': solved,
        }, f)

    print(f'training time: {time.time() - start_time:.2f}, '
          f'problem f{"" if solved else "not "}solved and data has been saved to "{filename}"')
