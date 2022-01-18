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


if __name__ == '__main__':
    # sim interact with env
    envs_n = 3
    envs = SubprocVecEnv([make_env("Pendulum-v0", i) for i in range(envs_n)])
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

    buffer = Buffer(10, 5)

    state = envs.reset()  # todo: how to handle next state
    buffer.states[0] = state  # initial state
    episode_reward = [0] * envs_n
    for _ in range(100):
        with torch.no_grad():
            action, action_log_prob = ac.act(state)
            state_value = ac.evaluate(state)

        state, reward, dones, infos = envs.step(action)
        episode_reward = [a + b for a, b in zip(episode_reward, reward)]
        for i, done in enumerate(dones):
            if done:
                episode_reward[i] = 0

        buffer.add(state, state_value.detach().numpy(), action, action_log_prob.detach().numpy(), reward, dones)
        # NOTE that we are storing `next_state` and `current_state_value`

    with torch.no_grad():
        state_value = ac.evaluate(state)
    buffer.state_values[-1] = state_value.detach().numpy()  # last state value

    buffer.compute_gae()
    i = 1
    for batch in buffer.data_generator():
        print(f'batch: {i}, states size: {batch[0].shape}')
        i += 1
        # print(batch)

    buffer.reset()
