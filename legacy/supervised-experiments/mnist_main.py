import argparse
import datetime
import os
import pickle
import platform

import gym
import IPython
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from context import es
from es.eval_funs import supervised_eval, supervised_test
from es.models import MNISTNet
from es.train import train_loop
from es.utils import get_inputs_from_args, load_checkpoint
from torchvision import datasets, transforms


if __name__ == '__main__':
    # Parse inputs
    parser = argparse.ArgumentParser(description='ES')
    parser.add_argument('--env-name', type=str, default='MNISTNet', metavar='ENV', help='environment')
    parser.add_argument('--model', type=str, default='FFN', choices=['DQN', 'FFN', 'Mujoco', 'ES'], metavar='MOD', help='model name')

    parser.add_argument('--optimizer', type=str, default='SGD', help='optimizer to use')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='optimizer learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='optimizer momentum')
    parser.add_argument('--nesterov', action='store_true', help='boolean to denote if optimizer momentum is Nesterov')
    parser.add_argument('--weight-decay', type=float, default=0.001, help='optimizer L2 norm weight decay penalty')
    
    parser.add_argument('--lr-scheduler', type=str, default='ExponentialLR', help='learning rate scheduler')
    parser.add_argument('--gamma', type=float, default=1, help='learning rate decay rate')
    parser.add_argument('--factor', type=float, default=0.8, help='reduction factor [ReduceLROnPlateau]')
    parser.add_argument('--patience', type=int, default=100, help='patience before lowering learning rate [ReduceLROnPlateau]')
    parser.add_argument('--threshold', type=float, default=1e-4, help='threshold for comparing best to current [ReduceLROnPlateau]')
    parser.add_argument('--cooldown', type=int, default=50, help='cooldown after lowering learning rate before able to do it again [ReduceLROnPlateau]')
    parser.add_argument('--mode', type=str, default='max', help='the optimization mode (minimization or maximization) [ReduceLROnPlateau]')
    parser.add_argument('--min-lr', type=float, default=1e-6, help='minimal learning rate [ReduceLROnPlateau]')
    parser.add_argument('--milestones', type=list, default=50, help='milestones on which to lower learning rate[MultiStepLR]')
    parser.add_argument('--step-size', type=int, default=50, help='step interval on which to lower learning rate[StepLR]')

    parser.add_argument('--agents', type=int, default=40, metavar='N', help='number of children, must be even')
    parser.add_argument('--sigma', type=float, default=0.05, metavar='SD', help='initial noise standard deviation')
    parser.add_argument('--optimize-sigma', action='store_true', help='boolean to denote whether or not to optimize sigma')
    parser.add_argument('--safe-mutation', action='store_true', help='boolean to denote whether or not to use safe mutations')
    parser.add_argument('--batch-size', type=int, default=200, metavar='BS', help='batch size agent evaluation (max episode steps for RL setting rollouts)')
    parser.add_argument('--max-generations', type=int, default=7500, metavar='MG', help='maximum number of generations')

    parser.add_argument('--frame-size', type=int, default=84, metavar='FS', help='square size of frames in pixels')
    
    parser.add_argument('--test', action='store_true', help='Test the model (accuracy or env render), no training')
    parser.add_argument('--restore', default='', metavar='RES', help='checkpoint from which to restore')
    parser.add_argument('--cuda', action='store_true', default=False, help='enables CUDA training')
    parser.add_argument('--silent', action='store_true', help='Silence print statements during training')
    args = parser.parse_args()

    # Input validation
    assert args.agents % 2 == 0                                         # Even number of agents
    assert not args.test or (args.test and args.restore)                # Testing requires restoring a model
    assert not args.cuda or (args.cuda and torch.cuda.is_available())   # Can only use CUDA if avaiable
    
    # Create model
    args.model = MNISTNet()
    if args.cuda:
        args.model = args.model.cuda()

    # Parameters to optimize are model parameters that require gradient and sigma if chosen
    opt_pars = []
    opt_pars.append({'label': 'model_params', 'params': args.model.parameters()})
    if args.optimize_sigma:
        # Parameterize the variance to ensure sigma>0: sigma = 0.5*exp(beta)
        args.beta = Variable(torch.Tensor([np.log(2*args.sigma**2)]), requires_grad=True)
        opt_pars.append({'label': 'beta', 'params': args.beta, 'lr': args.lr/10})

    # Create optimizer
    OptimizerClass = getattr(optim, args.optimizer)
    optimizer_input_dict = es.utils.get_inputs_from_args(OptimizerClass.__init__, args)
    args.optimizer = OptimizerClass(opt_pars, **optimizer_input_dict)

    # Create learning rate scheduler
    SchedulerClass = getattr(optim.lr_scheduler, args.lr_scheduler)
    scheduler_input_dict = es.utils.get_inputs_from_args(SchedulerClass.__init__, args)
    if SchedulerClass is optim.lr_scheduler.ReduceLROnPlateau:
        scheduler_input_dict['mode'] = 'max'
    args.lr_scheduler = SchedulerClass(**scheduler_input_dict)

    # Set number of OMP threads for CPU computations
    # NOTE: This is needed for my personal stationary Linux PC for partially unknown reasons
    if platform.system() == 'Linux':
        torch.set_num_threads(1)

    # Create checkpoint directory if nonexistent
    file_path = os.path.split(os.path.realpath(__file__))[0]
    if not args.restore:
        timestamp = datetime.datetime.now().strftime("%y%m%d-%H%M%S.%f")
        args.chkpt_dir = os.path.join(file_path, 'checkpoints', '{:s}-{:s}'.format(args.env_name, timestamp))
        if not os.path.exists(args.chkpt_dir):
            os.makedirs(args.chkpt_dir)
    
    # Load checkpoint if specified
    stats = None
    if args.restore:
        args.chkpt_dir, args.model, args.optimizer, args.lr_scheduler, stats = load_checkpoint(args.restore, file_path, args.model, args.optimizer, args.lr_scheduler, args.test)

    # Training and test data loaders
    data_cuda_kwargs = {'num_workers': 1, 'pin_memory': True} if False else {}
    args.train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(os.path.join(file_path,'data'), train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **data_cuda_kwargs)
    args.test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(os.path.join(file_path,'data'), train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=1000, shuffle=True, **data_cuda_kwargs)

    # Run test or train
    if args.test:
        supervised_test(args, args.model, args.test_loader)
    else:
        try:
            train_loop(args, args.model, args.train_loader, supervised_eval, args.optimizer, args.lr_scheduler, stats=stats)
        except KeyboardInterrupt:
            print("Training stopped by user.")
        supervised_test(args.model, args.test_loader, cuda=args.cuda)
