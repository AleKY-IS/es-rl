from __future__ import absolute_import, division, print_function

import argparse
import os

import gym
import IPython
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.autograd import Variable

from context import es
from es.envs import create_atari_env
from es.eval_funs import gym_render, gym_rollout
from es.models import DQN, FFN
from es.train import train_loop

# Parse inputs
parser = argparse.ArgumentParser(description='ES')
parser.add_argument('--env-name', default='CartPole-v0', metavar='ENV', help='environment')
parser.add_argument('--model-name', default='FFN', choices=['DQN', 'FFN', 'Mujoco', 'ES'], metavar='MOD', help='model name')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate')
parser.add_argument('--lr-decay', type=float, default=1, metavar='LRD', help='learning rate decay')
parser.add_argument('--sigma', type=float, default=0.05, metavar='SD', help='noise standard deviation')
parser.add_argument('--useAdam', action='store_true', help='bool to determine if to use adam optimizer')
parser.add_argument('--n', type=int, default=40, metavar='N', help='batch size, must be even')
parser.add_argument('--max-episode-length', type=int, default=10000, metavar='MEL', help='maximum length of an episode')
parser.add_argument('--max-gradient-updates', type=int, default=100000, metavar='MGU', help='maximum number of updates')
parser.add_argument('--frame-size', type=int, default=84, metavar='FS', help='square size of frames in pixels')
parser.add_argument('--variable-ep-len', action='store_true', help="Change max episode length during training")
parser.add_argument('--silent', action='store_true', help='Silence print statements during training')
parser.add_argument('--test', action='store_true', help='Just render the env, no training')
parser.add_argument('--restore', default='', metavar='RES', help='checkpoint from which to restore')


if __name__ == '__main__':
    args = parser.parse_args()
    assert args.n % 2 == 0

    # Create environment
    if args.env_name == 'CartPole-v0' or args.env_name == 'CartPole-v1':
        env = gym.make(args.env_name)
    else:
        env = create_atari_env(args.env_name, square_size=args.frame_size)
        
    # Create model
    model_class = getattr(es.models, args.model_name)
    parent_model = model_class(env.observation_space, env.action_space)

    # Remove gradient requirement from parameters
    # for param in parent_model.parameters():
    #    param.requires_grad = False
    
    # Create checkpoint directory if nonexistent
    chkpt_dir = 'checkpoints/%s/' % args.env_name
    if not os.path.exists(chkpt_dir):
        os.makedirs(chkpt_dir)
    
    # Load checkpoint if specified
    if args.restore:
        try:
            #file_path = os.path.split(os.path.realpath(__file__))[0]
            state_dict = torch.load(args.restore)
            parent_model.load_state_dict(state_dict)
        except Exception:
            print("Checkpoint restore failed")

    # Run test or train
    if args.test:
        gym_render(args, parent_model, env)
    else:
        try:
            train_loop(args, parent_model, env, gym_rollout, chkpt_dir)
        except KeyboardInterrupt:
            print("Program stopped by keyboard interruption")
