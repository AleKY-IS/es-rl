import IPython
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def softmax(x, dim):
    """
    Make-shift version of softmax that works on linux.
    """
    s = x.data.exp().sum()
    x.data = x.data.exp()/s
    return x

def log_softmax(x, dim):
    x.data = softmax(x, dim).data.log()
    return x


class AbstractESModel(nn.Module):
    """
    Abstract models class for models that are trained by evolutionary
    methods. The models have the .parameters() method, but here an additional
    .es_parameters() method is added. Normally, parameters are trained/not trained
    based on the .requires_grad bool. For ES this is not really analogous to being
    trained or not.
    """
    def count_parameters(self):
        count = 0
        for param in self.parameters():
            count += param.data.numpy().flatten().shape[0]
        return count

    def es_parameters(self):
        """
        The params that should be trained by ES (all of them)
        """
        return self.parameters()


class FFN(AbstractESModel):
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
        x = F.log_softmax(self.lin5(x), dim=1)
        #x = F.relu(self.lin5(x))
        #x = log_softmax(self.lin5(x), dim=1)
        return x

class DQN(AbstractESModel):
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


class MujocoFFN(AbstractESModel):
    """
    The FFN used by Salismans (2017)
    """
    def __init__(self, observation_space, action_space):
        super(MujocoFFN, self).__init__()
        self.full1 = nn.Linear(observation_space.shape, 2)

    def forward(self, x):
        x = x
        return None


class MNISTNet(AbstractESModel):
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
