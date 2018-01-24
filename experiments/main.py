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
from es.eval_funs import gym_rollout, gym_test, gym_render, supervised_eval, supervised_test
from es.models import *
from es.train import train_loop
from es.utils import get_inputs_from_args, load_checkpoint
from torchvision import datasets, transforms


supervised_datasets = ['MNIST', 'FashionMNIST']
nets = ['MNISTNet', 'DQN', 'FFN', 'Mujoco', 'ES']
algorithms = ['NES', 'xNES']


def parse_inputs():
    """
    Method to parse inputs given through the terminal.
    """
    parser = argparse.ArgumentParser(description='Experiments')
    # Environment
    parser.add_argument('--env-name', type=str, default='MNIST', metavar='ENV', help='environment')
    parser.add_argument('--frame-size', type=int, default=84, metavar='FS', help='square size of frames in pixels')
    # Model
    parser.add_argument('--model', type=str, default='MNISTNet', choices=nets, metavar='MOD', help='model name in es.models')
    # Optimizer
    parser.add_argument('--optimizer', type=str, default='SGD', help='optimizer to use')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='optimizer learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='optimizer momentum')
    parser.add_argument('--nesterov', action='store_true', help='boolean to denote if optimizer momentum is Nesterov')
    parser.add_argument('--weight-decay', type=float, default=0.001, help='optimizer L2 norm weight decay penalty')
    # Learning rate scheduler
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
    # Algorithm
    parser.add_argument('--algorithm', type=str, default='NES', metavar='ALG', help='model name in es.models')
    parser.add_argument('--agents', type=int, default=40, metavar='N', help='number of children, must be even')
    parser.add_argument('--sigma', type=float, default=0.05, metavar='SD', help='initial noise standard deviation')
    parser.add_argument('--optimize-sigma', action='store_true', help='boolean to denote whether or not to optimize sigma')
    parser.add_argument('--safe-mutation', action='store_true', help='boolean to denote whether or not to use safe mutations')
    parser.add_argument('--batch-size', type=int, default=200, metavar='BS', help='batch size agent evaluation (max episode steps for RL setting rollouts)')
    parser.add_argument('--max-generations', type=int, default=7500, metavar='MG', help='maximum number of generations')
    # Execution
    parser.add_argument('--test', action='store_true', help='Test the model (accuracy or env render), no training')
    parser.add_argument('--restore', default='', metavar='RES', help='checkpoint from which to restore')
    parser.add_argument('--cuda', action='store_true', default=False, help='enables CUDA training')
    parser.add_argument('--silent', action='store_true', help='Silence print statements during training')
    args = parser.parse_args()
    return args


def validate_inputs(args):
    # Input validation
    assert args.agents % 2 == 0                                         # Even number of agents
    assert not args.test or (args.test and args.restore)                # Testing requires restoring a model
    assert not args.cuda or (args.cuda and torch.cuda.is_available())   # Can only use CUDA if avaiable

    # Determine supervised/reinforcement learning problem
    if args.env_name in gym.envs.registry.env_specs.keys():
        args.is_supervised = False
        args.is_rl = True
    elif args.env_name in supervised_datasets:
        args.is_supervised = True
        args.is_rl = False
    else:
        raise ValueError('The given environment name is unrecognized')


def create_algorithma(args):
    model_class = getattr(es.algorithms, args.algorithm)


def create_model(args):
    # Create model
    model_class = getattr(es.models, args.model)
    # Supervised or RL
    if args.is_rl:
        args.model = model_class(args.env.observation_space, args.env.action_space)
    elif args.is_supervised:
        if args.env_name == 'MNIST':
            args.model = MNISTNet()
        elif args.env_name == 'FashionMNIST':
            args.model = MNISTNet()
    assert type(args.model) is not str
    # CUDA
    if args.cuda:
        args.model = args.model.cuda()


def create_optimizer(args):
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


def create_lr_scheduler(args):
    # Create learning rate scheduler
    SchedulerClass = getattr(optim.lr_scheduler, args.lr_scheduler)
    scheduler_input_dict = es.utils.get_inputs_from_args(SchedulerClass.__init__, args)
    # Set mode to maximization if the scheduler is `ReduceLRONPlateau`
    if SchedulerClass is optim.lr_scheduler.ReduceLROnPlateau:
        scheduler_input_dict['mode'] = 'max'
    args.lr_scheduler = SchedulerClass(**scheduler_input_dict)


def get_checkpoint(args):
    # Initialize args.stats
    args.stats = None
    if not args.restore:
        # Create checkpoint directory if not restoring
        timestamp = datetime.datetime.now().strftime("%y%m%d-%H%M%S.%f")
        args.chkpt_dir = os.path.join(args.file_path, 'checkpoints', '{:s}-{:s}'.format(args.env_name, timestamp))
        if not os.path.exists(args.chkpt_dir):
            os.makedirs(args.chkpt_dir)
    else:
        # Load checkpoint if restoring
        args.chkpt_dir, args.model, args.optimizer, args.lr_scheduler, args.stats = load_checkpoint(args.restore, args.file_path, args.model, args.optimizer, args.lr_scheduler, args.test)


def create_environment(args):
    if args.is_rl:
        if args.env_name == 'CartPole-v0' or args.env_name == 'CartPole-v1':
            args.env = gym.make(args.env_name)
        else:
            args.env = create_atari_env(args.env_name, square_size=args.frame_size)
    elif args.is_supervised:
        data_cuda_kwargs = {'num_workers': 1, 'pin_memory': True} if False else {}
        if args.env_name == 'MNIST':
            batch_size = args.batch_size if not(args.test) else 1000
            data_dir = os.path.join(args.file_path, 'data', 'MNIST')
            data_set = datasets.MNIST(data_dir,
                                      train=not(args.test),
                                      download=True,
                                      transform=transforms.Compose([
                                          transforms.ToTensor(),
                                          transforms.Normalize(
                                              (0.1307,), (0.3081,))
                                      ]))
            args.env = torch.utils.data.DataLoader(data_set,
                                                   batch_size=batch_size, 
                                                   shuffle=True, **data_cuda_kwargs)
        if args.env_name == 'FashionMNIST':
            batch_size = args.batch_size if not(args.test) else 1000
            data_dir = os.path.join(args.file_path, 'data', 'Fashion-MNIST')
            data_set = datasets.FashionMNIST(data_dir,
                                             train=not(args.test),
                                             download=True,
                                             transform=transforms.Compose([
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(
                                                     (0.1307,), (0.3081,))
                                             ]))
            args.env = torch.utils.data.DataLoader(data_set,
                                                   batch_size=batch_size, 
                                                   shuffle=True, **data_cuda_kwargs)
    assert hasattr(args, 'env')


def get_eval_funs(args):
    if args.is_rl:
        args.eval_fun = gym_rollout
        args.test_fun = gym_test
        args.rend_fun = gym_render
    elif args.is_supervised:
        args.eval_fun = supervised_eval
        args.test_fun = supervised_test


if __name__ == '__main__':
    # Parse and validate
    args = parse_inputs()
    validate_inputs(args)
    args.file_path = os.path.split(os.path.realpath(__file__))[0]
    # Create environment, model, optimizer and learning rate scheduler
    create_algorithm(args)
    create_environment(args)
    create_model(args)
    create_optimizer(args)
    create_lr_scheduler(args)
    # Create checkpoint
    get_checkpoint(args)
    # Get functions for evaluation and testing
    get_eval_funs(args)
    # Get input dictionary of args for algorithm
    args_dict = vars(args)
    input_dict = get_inputs_from_args(args.algorithm.__init__, args)

    # Set number of OMP threads for CPU computations
    # NOTE: This is needed for my personal stationary Linux PC for partially unknown reasons
    if platform.system() == 'Linux':
        torch.set_num_threads(1)  
    
    # Train if not testing
    if not args.test:
        try:
            train_loop(args, args.model, args.env, supervised_eval, args.optimizer, args.lr_scheduler, stats=args.stats)
        except KeyboardInterrupt:
            print("Training stopped by user.")
        # Remake the environment in test mode
        args.test = True
        create_environment(args)
    # Test
    supervised_test(args, args.model, args.test_loader)
