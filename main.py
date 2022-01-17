import gym
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv

from ActorCritic import ActorCritic


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

    obs = envs.reset()
    rewards = [0] * envs_n
    for _ in range(10):
        action, _ = ac.act(obs)
        state_value = ac.evaluate(obs)
        obs, reward, dones, infos = envs.step(action)
        rewards = [a + b for a, b in zip(rewards, reward)]
        # print(rewards)
        for i, done in enumerate(dones):
            if done:
                rewards[i] = 0
