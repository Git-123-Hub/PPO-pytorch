import math

import gym
import torch
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv

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

    # set up buffer
    buffer = Buffer(int(learn_interval // num_process), int(batch_size))

    learn_times = math.ceil(total_steps / learn_interval)
    for n in range(learn_times):
        # initial buffer and envs
        buffer.reset()
        state = envs.reset()  # todo: how to handle next state
        buffer.states[0] = state  # initial state
        episode_reward = [0] * num_process

        # interact with envs
        envs_steps = math.ceil(learn_interval / num_process)
        # each time learn, envs should run for these steps to fill the buffer, i.e. prepare the data
        for _ in range(envs_steps):
            with torch.no_grad():
                action, action_log_prob = ac.act(state)
                state_value = ac.evaluate(state)
            state, reward, dones, infos = envs.step(action)
            episode_reward = [a + b for a, b in zip(episode_reward, reward)]
            for i, done in enumerate(dones):
                if done:
                    episode_reward[i] = 0

            buffer.add(state, state_value.detach().numpy(), action, action_log_prob.detach().numpy(), reward,
                       dones)
            # NOTE that we are storing `next_state` and `current_state_value`

        with torch.no_grad():
            state_value = ac.evaluate(state)
        buffer.state_values[-1] = state_value.detach().numpy()  # last state value
        assert buffer.capacity == buffer.size, 'the buffer is not full'

        buffer.compute_gae()
        # start to learn
        # todo: ppo !!!!!!

        print(f'{"-" * 10} update {n + 1}, experience collected: {buffer.size * num_process} {"-" * 10}')
        batch_n = 0
        for batch in buffer.data_generator():
            batch_n += 1
            print(f'batch: {batch_n}, length: {len(batch[-1])}, indices: {batch[-1]}')
