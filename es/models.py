# Taken from https://github.com/ikostrikov/pytorch-a3c
from __future__ import absolute_import, division, print_function

import IPython
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1).expand_as(out))
    return out


def weights_init(m):
    """
    Not actually using this but let's keep it here in case that changes
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)


def selu(x):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * F.elu(x, alpha)


class DQN(nn.Module):
    """
    The CNN used by Mnih et al (2015)
    """
    def __init__(self, observation_space, action_space):
        super(DQN, self).__init__()
        in_channels = observation_space.shape[0]
        out_dim = action_space.n
        self.conv1 = nn.Conv2d(in_channels, out_channels=32, kernel_size=(8, 8), stride=(4, 4))
        self.conv2 = nn.Conv2d(32, out_channels=64, kernel_size=(4, 4), stride=(2, 2))
        self.conv3 = nn.Conv2d(64, out_channels=64, kernel_size=(3, 3), stride=(1, 1))
        n_size = self._get_conv_output(observation_space.shape)
        self.head = nn.Linear(n_size, out_dim)
        
        # in_channels = observation_space.shape[0]
        # self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=5, stride=2)
        # self.bn1 = nn.BatchNorm2d(16)
        # self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        # self.bn2 = nn.BatchNorm2d(32)
        # self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        # self.bn3 = nn.BatchNorm2d(32)
        # self.head = nn.Linear(448, 2)

    def forward(self, x):
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)
        x = F.softmax(self.head(x), dim=1)
        return x

        # x = F.relu(self.bn1(self.conv1(x)))
        # x = F.relu(self.bn2(self.conv2(x)))
        # x = F.relu(self.bn3(self.conv3(x)))

    def _get_conv_output(self, shape):
        """ Compute the number of output parameters from convolutional part by forward pass"""
        bs = 1
        input = Variable(torch.rand(bs, *shape))
        output_feat = self._forward_features(input)
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size

    def _forward_features(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x


class MujocoFFN(nn.Module):
    """
    The FFN used by Salismans (2017)
    """
    def __init__(self, observation_space, action_space):
        super(DQN, self).__init__()
        self.full1 = nn.Linear(observation_space.shape, 2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.head = nn.Linear(448, 2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


class ES(nn.Module):

    def __init__(self, num_inputs, action_space, small_net=False):
        """
        Really I should be using inheritance for the small_net here
        """
        super(ES, self).__init__()
        num_outputs = action_space.n
        self.small_net = small_net
        if self.small_net:
            self.linear1 = nn.Linear(num_inputs, 64)
            self.linear2 = nn.Linear(64, 64)
            self.actor_linear = nn.Linear(64, num_outputs)
        else:
            self.conv1 = nn.Conv2d(num_inputs, 32, kernel_size=3, stride=2, padding=1)
            self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
            self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
            self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
            self.lstm = nn.LSTMCell(32*3*3, 256)
            self.actor_linear = nn.Linear(256, num_outputs)
        self.train()


    def forward(self, inputs):
        if self.small_net:
            x = selu(self.linear1(inputs))
            x = selu(self.linear2(x))
            return self.actor_linear(x)
        else:
            inputs, (hx, cx) = inputs
            x = selu(self.conv1(inputs))
            x = selu(self.conv2(x))
            x = selu(self.conv3(x))
            x = selu(self.conv4(x))
            x = x.view(-1, 32*3*3)
            hx, cx = self.lstm(x, (hx, cx))
            x = hx
            return self.actor_linear(x), (hx, cx)

    def count_parameters(self):
        count = 0
        for param in self.parameters():
            count += param.data.numpy().flatten().shape[0]
        return count

    def es_params(self):
        """
        The params that should be trained by ES (all of them)
        """
        return [(k, v) for k, v in zip(self.state_dict().keys(),
                                       self.state_dict().values())]
