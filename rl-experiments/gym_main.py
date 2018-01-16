from __future__ import absolute_import, division, print_function

import argparse
import os
import platform
import datetime

import gym
import IPython
import torch
import torch.optim as optim
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.autograd import Variable

from context import es
from es.envs import create_atari_env
from es.eval_funs import gym_render, gym_rollout
from es.models import DQN, FFN
from es.train import train_loop


if __name__ == '__main__':
    # Parse inputs
    parser = argparse.ArgumentParser(description='ES')
    parser.add_argument('--env-name', type=str, default='CartPole-v0', metavar='ENV', help='environment')
    parser.add_argument('--model', type=str, default='FFN', choices=['DQN', 'FFN', 'Mujoco', 'ES'], metavar='MOD', help='model name')

    parser.add_argument('--optimizer', type=str, default='SGD', help='optimizer to use')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='optimizer learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='optimizer momentum')
    parser.add_argument('--nesterov', action='store_true', help='boolean to denote if optimizer momentum is Nesterov')
    parser.add_argument('--weight-decay', type=float, default=0.001, help='optimizer L2 norm weight decay penalty')

    parser.add_argument('--lr-scheduler', type=str, default='ReduceLROnPlateau', help='learning rate scheduler')
    parser.add_argument('--gamma', type=float, default=0.99, help='learning rate decay rate')
    parser.add_argument('--factor', type=float, default=0.1, help='reduction factor [ReduceLROnPlateau]')
    parser.add_argument('--patience', type=int, default=10, help='patience before lowering learning rate [ReduceLROnPlateau]')
    parser.add_argument('--threshold', type=float, default=1e-4, help='threshold for comparing best to current [ReduceLROnPlateau]')
    parser.add_argument('--cooldown', type=int, default=10, help='cooldown after lowering learning rate before able to do it again [ReduceLROnPlateau]')
    parser.add_argument('--min-lr', type=float, default=1e-6, help='minimal learning rate [ReduceLROnPlateau]')
    parser.add_argument('--milestones', type=list, default=50, help='milestones on which to lower learning rate[MultiStepLR]')
    parser.add_argument('--step-size', type=int, default=50, help='step interval on which to lower learning rate[StepLR]')

    parser.add_argument('--agents', type=int, default=40, metavar='N', help='number of children, must be even')
    parser.add_argument('--sigma', type=float, default=0.05, metavar='SD', help='initial noise standard deviation')
    parser.add_argument('--batch-size', type=int, default=10000, metavar='BS', help='batch size agent evaluation (max episode steps for RL setting rollouts)')
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
    
    # Create environment
    if args.env_name == 'CartPole-v0' or args.env_name == 'CartPole-v1':
        env = gym.make(args.env_name)
    else:
        env = create_atari_env(args.env_name, square_size=args.frame_size)
    
    # Create model
    model_class = getattr(es.models, args.model)
    model = model_class(env.observation_space, env.action_space)
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
    timestamp = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    chkpt_dir = 'checkpoints/{:s}-{:s}/'.format(args.env_name, timestamp)
    if not os.path.exists(chkpt_dir):
        os.makedirs(chkpt_dir)
    
    # Load checkpoint if specified
    stats = None
    if args.restore:
        try:
            file_path = os.path.split(os.path.realpath(__file__))[0]
            restore_path = file_path+'/'+'/'.join([i for i in args.restore.split('/') if i not in file_path.split('/')])
            IPython.embed()
            model_state_dict = torch.load(os.path.join(restore_path, 'model_state_dict.pth'))
            optimizer_state_dict = torch.load(os.path.join(restore_path, 'optimizer_state_dict.pth'))
            with open(os.path.join(restore_path, 'stats.pkl'), 'rb') as filename:
                stats = pickle.load(filename)
            lr_scheduler.last_epoch = stats['generations'][-1]
            model.load_state_dict(model_state_dict)
            optimizer.load_state_dict(optimizer_state_dict)
        except Exception:
            print("Checkpoint restore failed")

    # Run test or train
    if args.test:
        gym_render(args, model, env)
    else:
        try:
            train_loop(args, model, env, gym_rollout, optimizer, lr_scheduler, chkpt_dir, stats=stats)
        except KeyboardInterrupt:
            print("Program stopped by keyboard interruption")
