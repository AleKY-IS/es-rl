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
        elif self.acti == 'relu':
            x = F.relu(self.lin5(x))
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
        action = actions.max(1)[1].data
        # Step
        state, reward, done, _ = env.step(action[0])
        retrn += reward
        nsteps += 1
        # Cast state
        state = Variable(torch.from_numpy(state).float(),
                         requires_grad=True).unsqueeze(0)


for acti in ['relu', 'softmax']:
    ts = time.clock()
    n = 10000
    env = gym.make('CartPole-v0')
    #env = create_atari_env('Freeway-v0')
    model = FFN(env.observation_space, env.action_space, acti)
    
    for i in range(n):
        gym_rollout(1000, model, 'dummy_seed', env, 'dummy_neg')

    tf = time.clock()
    print(acti + " total: " + str(tf-ts))
    print(acti + " per call: " + str((tf - ts) / n))
