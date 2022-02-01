import argparse


def get_args():
    parser = argparse.ArgumentParser(description='PPO')

    parser.add_argument('--seed', type=int, default=None, help='random seed (default: None)')
    parser.add_argument('--env-name', type=str, default='Pendulum-v0', help='name of the environment')
    parser.add_argument('--folder', type=str, default=None,
                        help='folder to save fig and data during training')
    parser.add_argument('--total-steps', type=int, default=300000,
                        help='total steps that interact with the environment during training')
    parser.add_argument('--num-process', type=int, default=4, help='num of envs that run in parallel')
    parser.add_argument('--learn-interval', type=int, default=2400,
                        help='step interval between each learning')
    parser.add_argument('--batch-size', type=int, default=600,
                        help='number of experiences sampled for buffer each time learning')
    parser.add_argument('--learn-epoch', type=int, default=30, help='epoch of learning when ppo update')
    parser.add_argument('--clip-ratio', type=float, default=0.5, help='ppo clip parameter')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                        help='learning rate for actor-critic (initial value for learning rate if --lr-decay)')
    parser.add_argument('--clip-grad', type=float, default=0.5, help='clip gradient norm')
    parser.add_argument('--lr-decay', action='store_true',
                        help='decay learning rate of optimizer linearly during training')
    parser.add_argument('--std-decay', action='store_true',
                        help='decay std of action distribution linearly during training')
    # todo: linear decay: lr and std for ac; hidden-layer;

    args = parser.parse_args()
    return args
