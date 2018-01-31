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


def capsule_softmax(input, dim=1):
    transposed_input = input.transpose(dim, len(input.size()) - 1)
    softmaxed_output = F.softmax(transposed_input.contiguous().view(-1, transposed_input.size(-1)))
    return softmaxed_output.view(*transposed_input.size()).transpose(dim, len(input.size()) - 1)


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


class CapsuleLayer(nn.Module):
    def __init__(self, num_capsules, num_route_nodes, in_channels, out_channels, kernel_size=None, stride=None,
                 num_iterations=3):
        super(CapsuleLayer, self).__init__()
        self.num_route_nodes = num_route_nodes
        self.num_iterations = num_iterations
        self.num_capsules = num_capsules
        if num_route_nodes != -1:
            self.route_weights = nn.Parameter(torch.randn(num_capsules, num_route_nodes, in_channels, out_channels))
        else:
            self.capsules = nn.ModuleList(
                [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0) for _ in
                 range(num_capsules)])

    def squash(self, tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm)

    def forward(self, x):
        if self.num_route_nodes != -1:
            priors = x[None, :, :, None, :] @ self.route_weights[:, None, :, :, :]
            logits = Variable(torch.zeros(*priors.size())).cuda()
            for i in range(self.num_iterations):
                probs = capsule_softmax(logits, dim=2)
                outputs = self.squash((probs * priors).sum(dim=2, keepdim=True))
                if i != self.num_iterations - 1:
                    delta_logits = (priors * outputs).sum(dim=-1, keepdim=True)
                    logits = logits + delta_logits
        else:
            outputs = [capsule(x).view(x.size(0), -1, 1) for capsule in self.capsules]
            outputs = torch.cat(outputs, dim=-1)
            outputs = self.squash(outputs)
        return outputs


class CapsuleNet(nn.Module):
    def __init__(self, n_classes):
        super(CapsuleNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=9, stride=1)
        self.primary_capsules = CapsuleLayer(num_capsules=8, num_route_nodes=-1, in_channels=256, out_channels=32,
                                             kernel_size=9, stride=2)
        self.digit_capsules = CapsuleLayer(num_capsules=n_classes, num_route_nodes=32 * 6 * 6, in_channels=8,
                                           out_channels=16)
        self.decoder = nn.Sequential(
            nn.Linear(16 * n_classes, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 784),
            nn.Sigmoid()
        )

    def forward(self, x, y=None):
        x = F.relu(self.conv1(x), inplace=True)
        x = self.primary_capsules(x)
        x = self.digit_capsules(x).squeeze().transpose(0, 1)
        classes = (x ** 2).sum(dim=-1) ** 0.5
        classes = F.softmax(classes)
        if y is None:
            # In all batches, get the most active capsule.
            _, max_length_indices = classes.max(dim=1)
            y = Variable(torch.sparse.torch.eye(n_classes)).cuda().index_select(dim=0, index=max_length_indices.data)
        reconstructions = self.decoder((x * y[:, :, None]).view(x.size(0), -1))
        return classes, reconstructions


class CapsuleLoss(nn.Module):
    def __init__(self):
        super(CapsuleLoss, self).__init__()
        self.reconstruction_loss = nn.MSELoss(size_average=False)

    def forward(self, images, labels, classes, reconstructions):
        left = F.relu(0.9 - classes, inplace=True) ** 2
        right = F.relu(classes - 0.1, inplace=True) ** 2
        margin_loss = labels * left + 0.5 * (1. - labels) * right
        margin_loss = margin_loss.sum()
        reconstruction_loss = self.reconstruction_loss(reconstructions, images)
        return (margin_loss + 0.0005 * reconstruction_loss) / images.size(0)
