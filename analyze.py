import argparse
import os
import pickle

import matplotlib.pyplot as plt


def analyze(folder):
    """plot all the running rewards of ppo training result on the same figure"""
    assert os.path.exists(folder), f'folder {folder} is not exist'

    fig, ax = plt.subplots()
    ax.set_xlabel('episode')
    ax.set_ylabel('running reward')

    env_name, goal = None, None
    for file in os.listdir(folder):
        filename, file_extension = os.path.splitext(file)
        if file_extension != '.pkl':
            continue
        with open(os.path.join(folder, file), 'rb') as f:
            data = pickle.load(f)
            if env_name is None:
                env_name = data['args'].env_name
                goal = data['goal']
            else:
                assert env_name == data['args'].env_name, "these data are not solving the same problem"

            running_rewards = data['running_rewards']
            x = range(1, len(running_rewards) + 1)
            ax.plot(x, running_rewards)

    # ax.hlines(y=goal, xmin=1, xmax=x[-1], color='red')  # plot goal of the environment

    name = f'running reward of PPO solve {env_name}'
    ax.set_title(name)
    plt.savefig(name)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='analyze')
    parser.add_argument('folder', type=str, help='folder where all the data are stored')
    args = parser.parse_args()
    analyze(args.folder)
