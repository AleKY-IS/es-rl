import math
import os
import queue
import time

import IPython
import matplotlib.pyplot as plt
import numpy as np

import gym
import torch
import torch.legacy.optim as legacyOptim
import torch.multiprocessing as mp
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

from context import es
from es.envs import create_atari_env


class FFN(nn.Module):
    """
    FFN for classical control problems
    """

    def __init__(self, observation_space, action_space, acti='relu'):
        super(FFN, self).__init__()
        assert hasattr(observation_space, 'shape') and len(
            observation_space.shape) == 1
        assert hasattr(action_space, 'n')
        in_dim = observation_space.shape[0]
        out_dim = action_space.n
        self.lin1 = nn.Linear(in_dim, 32)
        self.lin2 = nn.Linear(32, 64)
        self.lin3 = nn.Linear(64, 64)
        self.lin4 = nn.Linear(64, 32)
        self.lin5 = nn.Linear(32, out_dim)
        self.acti = acti

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        x = F.relu(self.lin4(x))
        if self.acti == 'softmax':
            x = F.log_softmax(self.lin5(x), dim=1)
            #print("did softmax")
        elif self.acti == 'relu':
            x = F.relu(self.lin5(x))
            #print("did relu")
        return x


class DQN(nn.Module):
    """
    The CNN used by Mnih et al (2015)
    """

    def __init__(self, observation_space, action_space):
        super(DQN, self).__init__()
        assert hasattr(observation_space, 'shape') and len(
            observation_space.shape) == 3
        assert hasattr(action_space, 'n')
        in_channels = observation_space.shape[0]
        out_dim = action_space.n
        self.conv1 = nn.Conv2d(in_channels, out_channels=32,
                               kernel_size=(8, 8), stride=(4, 4))
        self.conv2 = nn.Conv2d(32, out_channels=64,
                               kernel_size=(4, 4), stride=(2, 2))
        self.conv3 = nn.Conv2d(64, out_channels=64,
                               kernel_size=(3, 3), stride=(1, 1))
        n_size = self._get_conv_output(observation_space.shape)
        self.lin1 = nn.Linear(n_size, 512)
        self.lin2 = nn.Linear(512, out_dim)

    def forward(self, x):
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.lin1(x))
        x = F.softmax(self.lin2(x), dim=1)
        #x = log_softmax(self.lin2(x), dim=1)
        return x

    def _get_conv_output(self, shape):
        """ Compute the number of output parameters from convolutional part by forward pass"""
        bs = 1
        inputs = Variable(torch.rand(bs, *shape))
        output_feat = self._forward_features(inputs)
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size

    def _forward_features(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x


def gym_rollout(max_episode_length, model, random_seed, env, is_antithetic):
    """
    Function to do rollouts of a policy defined by `model` in given environment
    """
    # Reset environment
    state = env.reset()
    state = Variable(torch.from_numpy(state).float(),
                     requires_grad=True).unsqueeze(0)
    retrn = 0
    nsteps = 0
    done = False
    # Rollout
    while not done and nsteps < max_episode_length:
        # Choose action
        actions = model(state)
        action = actions.max(1)[1].data.numpy()
        # Step
        state, reward, done, _ = env.step(action[0])
        retrn += reward
        nsteps += 1
        # Cast state
        state = Variable(torch.from_numpy(state).float(),
                         requires_grad=True).unsqueeze(0)


for acti in ['relu', 'softmax']:
    ts = time.clock()
    n = 1000
    env = gym.make('CartPole-v0')
    #env = create_atari_env('Freeway-v0')
    return_queue = mp.Queue()
    model = FFN(env.observation_space, env.action_space, acti)

    for i in range(n):
        gym_rollout(1000, model, 'dummy_seed', env, 'dummy_neg')

    # Get results of finished processes
    #raw_output = [return_queue.get() for _ in range(return_queue.qsize())]
    tf = time.clock()
    print(acti + " total: " + str(tf-ts))
    print(acti + " per call: " + str((tf - ts) / n))
