from __future__ import absolute_import, division, print_function

import argparse
import os

import gym
import IPython
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

from context import es
from es.eval_funs import supervised_eval, supervised_test
from es.models import MNISTNet
from es.train import train_loop

if __name__ == '__main__':
    # Parse inputs
    parser = argparse.ArgumentParser(description='ES')
    parser.add_argument('--env-name', default='MNIST', metavar='ENV', help='environment')
    #parser.add_argument('--model-name', default='FFN', choices=['DQN', 'FFN', 'Mujoco', 'ES'], metavar='MOD', help='model name')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate')
    parser.add_argument('--lr-decay', type=float, default=1, metavar='LRD', help='learning rate decay')
    parser.add_argument('--sigma', type=float, default=0.05, metavar='SD', help='noise standard deviation')
    parser.add_argument('--useAdam', action='store_true', help='bool to determine if to use adam optimizer')
    parser.add_argument('--n', type=int, default=40, metavar='N', help='batch size, must be even')
    #parser.add_argument('--max-episode-length', type=int, default=10000, metavar='MEL', help='maximum length of an episode')
    parser.add_argument('--max-gradient-updates', type=int, default=100000, metavar='MGU', help='maximum number of updates')
    #parser.add_argument('--frame-size', type=int, default=84, metavar='FS', help='square size of frames in pixels')
    #parser.add_argument('--variable-ep-len', action='store_true', help="Change max episode length during training")
    parser.add_argument('--silent', action='store_true', help='Silence print statements during training')
    parser.add_argument('--test', action='store_true', help='Just render the env, no training')
    parser.add_argument('--restore', default='', metavar='RES', help='checkpoint from which to restore')
    parser.add_argument('--cuda', action='store_true', default=False, help='enables CUDA training')
    args = parser.parse_args()

    # Even batch size
    assert args.n % 2 == 0
    
    # CUDA wanted and available
    args.cuda = args.cuda and torch.cuda.is_available()
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    # Training and test data loaders
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.n, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=1000, shuffle=True, **kwargs)

    # Create model
    parent_model = MNISTNet()

    """
    IPython.embed()

    print(next(iter(train_loader)))
    (data, target) = next(iter(train_loader))
    print(data)
    print(target)
    
    if args.cuda:
        data, target = data.cuda(), target.cuda()
    data, target = Variable(data), Variable(target)
    output = parent_model(data)
    loss = -F.nll_loss(output, target)
    """
    
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
        supervised_test(args, parent_model, test_loader)
    else:
        train_loop(args, parent_model, train_loader, supervised_eval, chkpt_dir)
