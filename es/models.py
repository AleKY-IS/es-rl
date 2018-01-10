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


class FFN(nn.Module):
    """
    FFN for classical control problems
    """
    def __init__(self, observation_space, action_space):
        super(FFN, self).__init__()
        assert hasattr(observation_space, 'shape') and len(observation_space.shape) == 1
        assert hasattr(action_space, 'n')
        in_dim = observation_space.shape[0]
        out_dim = action_space.n
        self.lin1 = nn.Linear(in_dim, 32)
        self.lin2 = nn.Linear(32, 64)
        self.lin3 = nn.Linear(64, 64)
        self.lin4 = nn.Linear(64, 32)
        self.lin5 = nn.Linear(32, out_dim)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        x = F.relu(self.lin4(x))
        x = F.softmax(self.lin5(x), dim=1)
        return x

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


class DQN(nn.Module):
    """
    The CNN used by Mnih et al (2015)
    """
    def __init__(self, observation_space, action_space):
        super(DQN, self).__init__()
        assert hasattr(observation_space, 'shape') and len(observation_space.shape) == 3
        assert hasattr(action_space, 'n')
        in_channels = observation_space.shape[0]
        out_dim = action_space.n
        self.conv1 = nn.Conv2d(in_channels, out_channels=32, kernel_size=(8, 8), stride=(4, 4))
        self.conv2 = nn.Conv2d(32, out_channels=64, kernel_size=(4, 4), stride=(2, 2))
        self.conv3 = nn.Conv2d(64, out_channels=64, kernel_size=(3, 3), stride=(1, 1))
        n_size = self._get_conv_output(observation_space.shape)
        self.lin1 = nn.Linear(n_size, 512)
        self.lin2 = nn.Linear(512, out_dim)

    def forward(self, x):
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.lin1(x))
        x = F.softmax(self.lin2(x), dim=1)
        return x

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


class MujocoFFN(nn.Module):
    """
    The FFN used by Salismans (2017)
    """
    def __init__(self, observation_space, action_space):
        super(DQN, self).__init__()
        self.full1 = nn.Linear(observation_space.shape, 2)

    def forward(self, x):
        x = x
        return None


class MNISTNet(nn.Module):
    """ 
    Convolutional neural network for use on the MNIST data set
    """
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=(5, 5))
        #self.conv1_bn = nn.BatchNorm2d(10)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=(5, 5))
        #self.conv2_bn = nn.BatchNorm2d(20)
        #self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        #x = self.conv1_bn(F.relu(F.max_pool2d(self.conv1(x), 2)))
        #x = self.conv2_drop(self.conv2_bn(F.relu(F.max_pool2d(self.conv2(x), 2))))
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        #x = F.relu(F.dropout(self.fc1(x), training=self.training))
        x = F.relu(self.fc1(x))
        x = F.log_softmax(self.fc2(x), dim=1)
        return x

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
