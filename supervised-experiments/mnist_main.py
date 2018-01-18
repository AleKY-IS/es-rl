import argparse
import datetime
import os
import pickle
import platform

import gym
import IPython
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from context import es
from es.eval_funs import supervised_eval, supervised_test
from es.models import MNISTNet
from es.train import train_loop
from es.utils import get_inputs_from_args
from torchvision import datasets, transforms


if __name__ == '__main__':
    # Parse inputs
    parser = argparse.ArgumentParser(description='ES')
    parser.add_argument('--env-name', type=str, default='MNISTNet', metavar='ENV', help='environment')
    parser.add_argument('--model', type=str, default='FFN', choices=['DQN', 'FFN', 'Mujoco', 'ES'], metavar='MOD', help='model name')

    parser.add_argument('--optimizer', type=str, default='SGD', help='optimizer to use')
    parser.add_argument('--lr', type=float, default=0.02, metavar='LR', help='optimizer learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='optimizer momentum')
    parser.add_argument('--nesterov', action='store_true', help='boolean to denote if optimizer momentum is Nesterov')
    parser.add_argument('--weight-decay', type=float, default=0.001, help='optimizer L2 norm weight decay penalty')

    parser.add_argument('--lr-scheduler', type=str, default='ReduceLROnPlateau', help='learning rate scheduler')
    parser.add_argument('--gamma', type=float, default=0.99, help='learning rate decay rate')
    parser.add_argument('--factor', type=float, default=0.8, help='reduction factor [ReduceLROnPlateau]')
    parser.add_argument('--patience', type=int, default=50, help='patience before lowering learning rate [ReduceLROnPlateau]')
    parser.add_argument('--threshold', type=float, default=1e-4, help='threshold for comparing best to current [ReduceLROnPlateau]')
    parser.add_argument('--cooldown', type=int, default=25, help='cooldown after lowering learning rate before able to do it again [ReduceLROnPlateau]')
    parser.add_argument('--mode', type=str, default='max', help='the optimization mode (minimization or maximization) [ReduceLROnPlateau]')
    parser.add_argument('--min-lr', type=float, default=1e-6, help='minimal learning rate [ReduceLROnPlateau]')
    parser.add_argument('--milestones', type=list, default=50, help='milestones on which to lower learning rate[MultiStepLR]')
    parser.add_argument('--step-size', type=int, default=50, help='step interval on which to lower learning rate[StepLR]')

    parser.add_argument('--agents', type=int, default=40, metavar='N', help='number of children, must be even')
    parser.add_argument('--sigma', type=float, default=0.05, metavar='SD', help='initial noise standard deviation')
    parser.add_argument('--batch-size', type=int, default=200, metavar='BS', help='batch size agent evaluation (max episode steps for RL setting rollouts)')
    parser.add_argument('--max-generations', type=int, default=100000, metavar='MG', help='maximum number of generations')

    parser.add_argument('--frame-size', type=int, default=84, metavar='FS', help='square size of frames in pixels')
    
    parser.add_argument('--silent', action='store_true', help='Silence print statements during training')
    parser.add_argument('--test', action='store_true', help='Test the model (accuracy or env render), no training')
    parser.add_argument('--restore', default='', metavar='RES', help='checkpoint from which to restore')
    parser.add_argument('--cuda', action='store_true', default=False, help='enables CUDA training')
    args = parser.parse_args()

    # Input validation
    assert args.agents % 2 == 0                                              # Even number of agents
    assert not args.test or (args.test and args.restore)                # Testing requires restoring a model
    assert not args.cuda or (args.cuda and torch.cuda_is_available())   # Can only use CUDA if avaiable
    
    # Create model
    model = MNISTNet()
    args.model = model

    # Create optimizer
    OptimizerClass = getattr(optim, args.optimizer)
    optimizer_input_dict = es.utils.get_inputs_from_args(OptimizerClass.__init__, args)
    optimizer = OptimizerClass(model.parameters(), **optimizer_input_dict)
    args.optimizer = optimizer

    # Create learning rate scheduler
    SchedulerClass = getattr(optim.lr_scheduler, args.lr_scheduler)
    scheduler_input_dict = es.utils.get_inputs_from_args(SchedulerClass.__init__, args)
    lr_scheduler = SchedulerClass(**scheduler_input_dict)
    args.lr_scheduler = lr_scheduler

    # Set number of OMP threads for CPU computations
    # NOTE: This is needed for my personal stationary Linux PC for partially unknown reasons
    if platform.system() == 'Linux':
        torch.set_num_threads(1)
    print("Num threads = " + str(torch.get_num_threads()))

    # Create checkpoint directory if nonexistent
    if not args.restore:
        timestamp = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
        chkpt_dir = 'checkpoints/{:s}-{:s}/'.format(args.env_name, timestamp)
        if not os.path.exists(chkpt_dir):
            os.makedirs(chkpt_dir)
    
    # Load checkpoint if specified
    stats = None
    if args.restore:
        file_path = os.path.split(os.path.realpath(__file__))[0]
        model, optimizer, lr_scheduler, stats = load_checkpoint(args.restore, file_path, model, optimizer, lr_scheduler, args.test)

    # Training and test data loaders
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=1000, shuffle=True, **kwargs)

    # Run test or train
    if args.test:
        supervised_test(args, model, test_loader)
    else:
        try:
            train_loop(args, model, train_loader, supervised_eval, optimizer, lr_scheduler, chkpt_dir, stats=stats)
        except KeyboardInterrupt:
            print("Training stopped by user.")
        supervised_test(args, model, test_loader)
