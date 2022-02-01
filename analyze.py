import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np


def analyze(folder):
    """analyze the statistical performance based on all the training result in a specific folder"""
    assert os.path.exists(folder), f'folder {folder} is not exist'
    all_running_rewards = []
    env_name = None
    for file in os.listdir(folder):
        filename, file_extension = os.path.splitext(file)
        if file_extension != '.pkl':
            continue
        with open(os.path.join(folder, file), 'rb') as f:
            data = pickle.load(f)
            if env_name is None:
                env_name = data['args'].env_name
            else:
                assert env_name == data['args'].env_name, "these data are not solving the same problem"
            all_running_rewards.append(data['running_rewards'])

    running_rewards = np.array(all_running_rewards)

    fig, ax = plt.subplots()
    ax.set_xlabel('episode')
    ax.set_ylabel('running reward')

    x = np.arange(1, len(running_rewards[-1]) + 1)
    mean = running_rewards.mean(axis=0)
    std = running_rewards.std(axis=0)

    color = 'blue'
    ax.plot(x, mean, color=color, )
    ax.plot(x, mean + std, color=color, alpha=0.2)
    ax.plot(x, mean - std, color=color, alpha=0.2)
    ax.fill_between(x, y1=mean - std, y2=mean + std, color=color, alpha=0.1)
    ax.hlines(y=-165, xmin=1, xmax=x[-1], color='red')

    name = f'running reward of PPO solve {env_name}'
    ax.set_title(name)
    plt.savefig(name)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='analyze')
    parser.add_argument('folder', type=str, help='folder where all the data are stored')
    args = parser.parse_args()
    analyze(args.folder)
