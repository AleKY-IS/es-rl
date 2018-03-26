import collections
import copy
import csv
import datetime
import math
import os
import pickle
import platform
import pprint
import queue
import time
from abc import ABCMeta, abstractmethod
from functools import partial
from itertools import zip_longest

import gym
import IPython
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
from torch.autograd import Variable

from context import utils
from utils.misc import get_inputs_from_dict, to_numeric
from utils.plotting import plot_stats
from utils.progress import PoolProgress
from utils.torchutils import torch_summarize


def count_model_parameters(model, only_trainable=True):
    """ Count the number of parameters in a pytorch model
    """
    if only_trainable:
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        n = sum(p.numel() for p in model.parameters())
    return n


class Algorithm(object):
    """Abstract class for variational algorithms

    Attributes:
        model (torch.nn): A pytorch module model
        env (gym): A gym environment
        optimizer (torch.optim.Optimizer): A pytorch optimizer
        lr_scheduler (torch.optim.lr_scheduler): A pytorch learning rate scheduler
        perturbations (int): The number of perturbed models to evaluate
        batch_size (int): The number of observations for an evaluate (supervised)
        max_generations (int): The maximum number of generations to train for
        safe_mutation (str): The version of safe mutations to use. Valid options are `ABS`, `SUM` and `SO`
        no_antithetic (bool): If `True`, the algorithm samples without also taking the antithetic sample
        chktp_dir (str): The directory to use to save/load checkpoints. If not absolute, it will be appended without overlap to the path of this file when executing
        chkpt_int (int): The interval in seconds between checkpoint saves. If chkpt_int<=0, a checkpoint is made at every generation.
        cuda (bool): Boolean to denote whether or not to use CUDA
        silent (bool): Boolean to denote if executing should be silent (no terminal printing)
    """

    __metaclass__ = ABCMeta

    def __init__(self, model, env, optimizer, lr_scheduler, eval_fun, perturbations, batch_size, max_generations, safe_mutation, no_antithetic, workers=mp.cpu_count(), chkpt_dir=None, chkpt_int=600, track_parallel=False, cuda=False, silent=False):
        self.algorithm = self.__class__.__name__
        # Algorithmic attributes
        self.model = model
        self.env = env
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.eval_fun = eval_fun
        self.safe_mutation = safe_mutation
        self.no_antithetic = no_antithetic
        self.perturbations = perturbations
        self.batch_size = batch_size
        self.max_generations = max_generations
        # Execution attributes
        self.workers = workers if workers <= mp.cpu_count() else mp.cpu_count()
        self.track_parallel = track_parallel
        self.cuda = cuda
        self.silent = silent
        # Checkpoint attributes
        self.chkpt_dir = chkpt_dir
        self.chkpt_int = chkpt_int
        # Safe mutation sensitivities
        n_layers = 0
        for _ in self.model.parameters():
            n_layers += 1
        self.sensitivities = [None]*n_layers
        # Get inputs for sensitivity calculations by sampling from environment or
        # getting a batch from the data loader
        if hasattr(self.env, 'observation_space'):
            # Preallocate for collecting..
            s = self.env.reset()
            self.sens_inputs = torch.zeros((1000, *s.shape))
            # Sample from environment
            # self.sens_inputs[0,...] = torch.from_numpy(s)
            # for i in range(999):
            #     self.sens_inputs[i+1,...] = torch.from_numpy(self.env.observation_space.sample())
        elif type(self.env) is torch.utils.data.DataLoader:
            self.sens_inputs = next(iter(self.env))[0]
        if self.safe_mutation is None:
            self.sens_inputs = self.sens_inputs[0].view(1, *self.sens_inputs.size()[1:])
        # Attributes to exclude from the state dictionary
        self.exclude_from_state_dict = {'env', 'optimizer', 'lr_scheduler', 'model', 'stats', 'sens_inputs'}
        # Initialize dict for saving statistics
        self._base_stat_keys = {'generations', 'walltimes', 'workertimes', 'unp_rank'}
        self.stats = {key: [] for key in self._base_stat_keys}
        self._training_start_time = None

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def perturb_model(self, seed):
        pass

    @abstractmethod
    def compute_gradients(self, *kwargs):
        pass

    def _add_parameter_to_optimize(self, parameter_group, overwrite=False):
        """Adds a parameter group to be optimized.

        The parameter group can consists of an arbitrary number of parameters that 
        will all get the same learning rate. The group is added to the optimizer and
        the lr_scheduler is reinitialized with the same inputs on the updated optimizer.

        If the given parameter_group has a label key, then this function returns a reference
        to the parameter group in the optimizer.
        
        Args:
            parameter_group (dict): The parameter group
        """
        # If a parameter group is labelled, check if already exists in order to delete
        if overwrite and 'label' in parameter_group:
            if len(list(filter(lambda group: group['label'] == parameter_group['label'], self.optimizer.param_groups))) > 0:
                self.optimizer.param_groups = [g for g in self.optimizer.param_groups if 'label' not in g or g['label'] != parameter_group['label']]
        # Add parameter group to optimizer and reinitialize lr scheduler
        self.optimizer.add_param_group(parameter_group)            
        inputs = get_inputs_from_dict(self.lr_scheduler.__class__.__init__, vars(self.lr_scheduler))
        self.lr_scheduler = self.lr_scheduler.__class__(**inputs)
        # Return reference if parameter group is labelled
        if 'label' in parameter_group:
            return list(filter(lambda group: group['label'] == parameter_group['label'], self.optimizer.param_groups))[0]

    @staticmethod
    def unperturbed_rank(returns, unperturbed_return):
        """Computes the rank of the unperturbed model among the perturbations.
        
        Args:
            returns (np.array): Returns of evaluated perturbed models.
            unperturbed_return (float): Return of the unperturbed model.
        
        Returns:
            int: Rank of the unperturbed model among the perturbations
        """      
        return (returns > unperturbed_return).sum() + 1

    @staticmethod
    def fitness_shaping(returns):
        """Computes the fitness shaped returns.

        Performs the fitness rank transformation used for CMA-ES.
        Reference: Natural Evolution Strategies [2014]
        
        Args:
            returns (np.array): Returns of evaluated perturbed models.
        
        Returns:
            np.array: Shaped returns
        """
        assert type(returns) == np.ndarray
        n = len(returns)
        sorted_indices = np.argsort(-returns)
        u = np.zeros(n)
        for k in range(n):
            u[sorted_indices[k]] = np.max([0, np.log(n / 2 + 1) - np.log(k + 1)])
        return u / np.sum(u) - 1 / n

    @staticmethod
    def fitness_normalization(returns, unperturbed_return):
        returns = returns - unperturbed_return
        if not np.isnan(returns.std()):
            returns = returns/returns.std()
        else:
            assert (returns == returns).all()
        return returns
    
    @staticmethod
    def get_perturbation(size, sensitivities=None, cuda=False):
        """Draws a perturbation tensor of dimension `size` from a standard normal.

        If `sensitivities is not None`, then the perturbation is scaled by these.
        If sensitivities are given and `sensitivities.size() == size[1:]` then the `size[0]` is 
        intepreted as a number of samples each of which is to be scaled by the given sensitivities.
        """
        if type(size) in [tuple, list]:
            size = torch.Size(size)
        elif type(size) in [torch.Size, int]:
            pass
        else:
            raise TypeError("Input `size` must be of type `int`, `list`, `tuple` or `torch.Size` but was `{}`".format(type(size).__name__))
        if sensitivities is not None and sensitivities.size() == size[1:]:
            samples = size[0]
        assert sensitivities is None or sensitivities.size() == size or sensitivities.size() == size[1:], "Sensitivities must match size of perturbation"
        if cuda:
            eps = torch.cuda.FloatTensor(size)
        else:
            eps = torch.FloatTensor(size)
        eps.normal_(mean=0, std=1)
        if sensitivities is not None:
            if sensitivities.size() == size[1:]:
                for s in range(samples):
                    eps[s, ...] /= sensitivities      # Scale by sensitivities
                    if eps.numel() > 1:
                        eps /= eps.std()              # Rescale to unit variance
            else:
                eps /= sensitivities     # Scale by sensitivities
                if eps.numel() > 1:
                    eps /= eps.std()     # Rescale to unit variance
        return eps

    @staticmethod
    def scale_by_sensitivities(eps, sensitivities=None):
        if sensitivities is None:
            return eps
        if eps.numel() > 1:
            # Scale to unit variance
            std = eps.std()
            eps /= std
        # Scale by sensitivities
        eps /= sensitivities
        if eps.numel() > 1:
            # Rescale to unit variance
            eps /= eps.std()
            # Rescale to original variance
            eps *= std
        return eps


    def _vec2modelrepr(self, vec, only_trainable=True, yield_generator=True):
        """Converts a 1xN Tensor into a list of Tensors, each matching the 
        dimension of the corresponding parameter in the model.
        """
        print("_vec2modelrepr")
        IPython.embed()
        assert self.model.count_parameters(only_trainable=True) == vec.numel()
        if yield_generator:
            i = 0
            for p in self.model.parameters():
                j = i + p.numel()
                yield vec[i:j]
                i = j
        else:
            pars = []
            i = 0
            for p in self.model.parameters():
                j = i + p.numel()
                pars.append(vec[i:j])
                i = j
            return pars

    def _modelrepr2vec(self, modelrepr, only_trainable=True):
        """Converts a list of Tensors to a single 1xN Tensor.
        """
        vec = torch.zeros(self.model.count_parameters(only_trainable=True))
        i = 0
        for p in modelrepr:
            j = i + p.numel()
            vec[i:j] = p.data.view(-1) if type(p) is Variable else p.view(-1)
            i = j
        return vec

    def compute_sensitivities(self, inputs=None, do_normalize=True, do_numerical=True):
        """Computes the output-weight sensitivities of the model.
       
        Currently implements the SM-G-ABS and SM-G-SUM safe mutations.

        Reference: Safe Mutations for Deep and Recurrent Neural Networks through Output Gradients [2017]
        
        Args:
            inputs (np.array): Batch of inputs on which to backpropagate and estimate sensitivities
            do_normalize (bool, optional): Defaults to True. Rescale all sensitivities to be between 0 and 1
            do_numerical (bool, optional): Defaults to True. Make sure sensitivities are not numerically ill conditioned (close to zero)
        
        Raises:
            NotImplementedError: For the SO safe mutation
            ValueError: For an unrecognized safe mutation
        """ 
        # Forward pass on input batch
        if inputs is None:
            inputs = self.sens_inputs
        if type(inputs) is not Variable:
            inputs = Variable(inputs)
        if self.cuda:
            inputs = inputs.cuda()
            self.model.cuda()
        if self.safe_mutation is None:
            # Dummy backprop to initialize gradients, then return
            output = self.model(inputs[0].view(1,*inputs[0].size()))
            if self.cuda:
                t = torch.cuda.FloatTensor(1, output.data.size()[1]).fill_(0)
            else:
                t = torch.zeros(1, output.data.size()[1])
            t[0,0] = 1
            output.backward(t)
            self.model.zero_grad()
            return
        outputs = self.model(inputs)
        batch_size = outputs.data.size()[0]
        n_outputs = outputs.data.size()[1]
        if self.cuda:
            t = torch.cuda.FloatTensor(batch_size, n_outputs).fill_(0)
        else:
            t = torch.zeros(batch_size, n_outputs)
        # Compute sensitivities using specified method
        if self.safe_mutation == 'ABS':
            sensitivities = self._compute_sensitivities_abs(outputs, t)
        elif self.safe_mutation == 'SUM':
            sensitivities = self._compute_sensitivities_sum(outputs, t)
        elif self.safe_mutation == 'SO':
            raise NotImplementedError('The second order safe mutation (SM-SO) is not yet implemented')
        elif self.safe_mutation == 'R':
            raise NotImplementedError('The SM-R safe mutation is not yet implemented')
        else:
            raise ValueError('The type ''{:s}'' of safe mutations is unrecognized'.format(self.safe_mutation))
        # IPython.embed()
        # Remove infs
        if do_numerical:
            overflow = False
            for pid in range(len(sensitivities)):
                sensitivities[pid][np.isinf(sensitivities[pid])] = 1
                overflow = overflow or np.isinf(sensitivities[pid]).any()
            if overflow: print('| Encountered numerical overflow in sensitivities', end='')
        # Normalize
        if do_normalize:
            # Find maximal sensitivity across all layers
            m = 0
            for pid in range(len(sensitivities)):
                m = np.max([m, sensitivities[pid].max()])
            if m == 0:
                print(' | All sensitivities were zero.', end='')
                for pid in range(len(sensitivities)):
                    sensitivities[pid] = None
                self.sensitivities = sensitivities
                return
            else:
                # Divide all layers by max (and clamp below and above)
                for pid in range(len(sensitivities)):
                    sensitivities[pid] /= m
                    sensitivities[pid].clamp_(min=1e-2, max=1)
        # Set sensitivities and assert their values
        for sens in sensitivities:
            assert not np.isnan(sens).any()
            assert not np.isinf(sens).any()
        self.sensitivities = sensitivities

    def _compute_sensitivities_abs(self, outputs, t):
        # Backward pass for each output unit (and accumulate gradients)
        sensitivities = []
        for k in range(t.size()[1]):
            self.model.zero_grad()
            # Compute dy_t/dw on batch
            for i in range(t.size()[0]):
                t.fill_(0)
                t[i, k] = 1
                outputs.backward(t, retain_graph=True)
                for param in self.model.parameters():
                    param.grad.data = param.grad.data.abs()
            # Get computed sensitivities and sum into those of other output units
            for pid, param in enumerate(self.model.parameters()):
                sens = param.grad.data.clone()  # Clone to sum correctly
                sens = sens.div(t.size()[0]).pow(2)
                if k == 0:
                    sensitivities.append(sens)
                else:
                    sensitivities[pid] += sens
        for pid, _ in enumerate(sensitivities):
            sensitivities[pid] = sensitivities[pid].sqrt()
        return sensitivities

    def _compute_sensitivities_sum(self, outputs, t):
        sensitivities = []
        # Backward pass for each output unit (and accumulate gradients)
        for k in range(t.size()[1]):
            self.model.zero_grad()
            # Compute dy_t/dw on batch
            # Sum over signed gradients
            t.fill_(0)
            t[:, k].fill_(1)
            outputs.backward(t, retain_graph=True)
            # Get computed sensitivities and sum into those of other output units
            for pid, param in enumerate(self.model.parameters()):
                sens = param.grad.data.clone()  # Clone to sum correctly
                sens = sens.pow(2)
                if k == 0:
                    sensitivities.append(sens)
                else:
                    sensitivities[pid] += sens
        for pid, _ in enumerate(sensitivities):
            sensitivities[pid] = sensitivities[pid].sqrt()
        return sensitivities

    def _compute_sensitivities_so(self, outputs, t):
        pass
    
    def _compute_sensitivities_r(self, outputs, t):
        pass

    def print_init(self):
        """Print the initial message when training is started
        """
        # Get strings to print
        env_name = self.env.spec.id if hasattr(self.env, 'spec') else self.env.dataset.root.split('/')[-1]
        safe_mutation = self.safe_mutation if self.safe_mutation is not None else 'None'
        # Build init string
        s = "=================== SYSTEM ====================\n"
        s += "System                {:s}\n".format(platform.system())
        s += "Machine               {:s}\n".format(platform.machine())
        s += "Platform              {:s}\n".format(platform.platform())
        s += "Platform version      {:s}\n".format(platform.version())
        s += "Processor             {:s}\n".format(platform.processor())
        s += "Available CPUs        {:s}\n".format(str(mp.cpu_count()))
        s += "\n==================== MODEL ====================\n"
        s += "Summary of " + self.model.__class__.__name__ + "\n\n"
        self.model.eval()
        model_summary = torch_summarize(input_size=self.sens_inputs[0].size(), model=self.model)
        s += model_summary.to_string() + "\n\n"
        s += "Parameters: {:d}".format(model_summary.n_parameters.sum()) + "\n"
        s += "Trainable parameters: {:d}".format(model_summary.n_trainable.sum()) + "\n"
        s += "Layers: {:d}".format(model_summary.shape[0]) + "\n"
        s += "Trainable layers: {:d}".format((model_summary.n_trainable != 0).sum()) + "\n"
        s += "\n================== OPTIMIZER ==================\n"
        s += str(type(self.optimizer)) + "\n"
        s += pprint.pformat(self.optimizer.state_dict()['param_groups']) + "\n"
        s += "\n================= LR SCHEDULE =================\n"
        s += str(type(self.lr_scheduler)) + "\n"
        s += pprint.pformat(vars(self.lr_scheduler)) + "\n"
        s += "\n================== ALGORITHM ==================\n"
        s += "Algorithm             {:s}\n".format(self.__class__.__name__)
        s += "Environment           {:s}\n".format(env_name)
        s += "Perturbations         {:d}\n".format(self.perturbations)
        s += "Generations           {:d}\n".format(self.max_generations)
        s += "Batch size            {:<5d}\n".format(self.batch_size)
        s += "Safe mutation         {:s}\n".format(safe_mutation)
        s += "Antithetic sampling   {:s}\n".format(str(not self.no_antithetic))
        s += "CUDA                  {:s}\n".format(str(self.cuda))
        s += "Workers               {:d}\n".format(self.workers)
        s += "Checkpoint interval   {:d}s\n".format(self.chkpt_int)
        s += "Checkpoint directory  {:s}\n".format(self.chkpt_dir)
        if self.chkpt_dir is not None:
            with open(os.path.join(self.chkpt_dir, 'init.log'), 'a') as f:
                f.write(s)
        if not self.silent:
            print(s, end='')

    def print_iter(self):
        """Print information on a generation during training.
        """
        if self.silent:
            return
        try:
            # O = 'O {1:' + str(len(str(self.batch_size * self.max_generations * self.perturbations))) + 'd}'
            G = 'G {0:' + str(len(str(self.max_generations))) + 'd}'
            R = 'R {5:' + str(len(str(self.perturbations))) + 'd}'
            s = '\n' + G + ' | F {1:7.2f} | A {2:7.2f} | Ma {3:7.2f} | Mi {4:7.2f} | ' + R + ' | L {6:5.4f}'
            s = s.format(self.stats['generations'][-1], self.stats['return_unp'][-1],
                         self.stats['return_avg'][-1],  self.stats['return_max'][-1],
                         self.stats['return_min'][-1],  self.stats['unp_rank'][-1],
                         self.stats['lr_0'][-1])
            if 'accuracy_unp' in self.stats.keys():
                s += ' | C {0:5.1f}%'.format(self.stats['accuracy_unp'][-1]*100)
            print(s, end='')
        except Exception:
            print('Could not print_iter', end='')

    def load_checkpoint(self, chkpt_dir, load_best=False, load_algorithm=False):
        """Loads a saved checkpoint.
        
        Args:
            chkpt_dir (str): Path to the checkpoint directory
            load_best (bool, optional): Defaults to False. Denotes whether or not to load the best model encountered. Otherwise the latest will be loaded.
        
        Raises:
            IOError: If the loading fails, an IOError exception is raised
        """
        # Get stats
        with open(os.path.join(chkpt_dir, 'stats.csv'), 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            stats = {k: [to_numeric(v)] for k, v in next(reader).items()}
            for record in reader:
                for k, v in record.items():
                    stats[k].append(to_numeric(v))
            stats.pop('')
        # Load state dict files
        algorithm_file = 'state-dict-best-algorithm.pkl' if load_best else 'state-dict-algorithm.pkl'
        model_file = 'state-dict-best-model.pkl' if load_best else 'state-dict-model.pkl'
        optimizer_file = 'state-dict-best-optimizer.pkl' if load_best else 'state-dict-optimizer.pkl'
        algorithm_state_dict = torch.load(os.path.join(chkpt_dir, algorithm_file))
        model_state_dict = torch.load(os.path.join(chkpt_dir, model_file))
        optimizer_state_dict = torch.load(os.path.join(chkpt_dir, optimizer_file))
        # Load state dicts into objects
        if load_algorithm:
            # Load algorithm state
            self.load_state_dict(algorithm_state_dict)
            # Load added parameters
            self._load_added_parameters(stats, optimizer_state_dict)
        self.chkpt_dir = chkpt_dir
        self.model.load_state_dict(model_state_dict)
        self.optimizer.load_state_dict(optimizer_state_dict)
        self.lr_scheduler.last_epoch = stats['generations'][-1]
        # Set constants
        self._training_start_time = algorithm_state_dict['_training_start_time'] + (time.time() - (stats['walltimes'][-1] + algorithm_state_dict['_training_start_time']))
        self._max_unp_return = np.max(stats['return_unp'])
        # self._max_unp_return = m
        for k in stats.keys(): self.stats[k] = []

    def _load_added_parameters(self, stats, optimizer_state_dict):
        for p in optimizer_state_dict['param_groups']:
            if 'label' in p and p['label'] != 'model_params':
                assert hasattr(self, p['label'])
                assert type(getattr(self, p['label'])) is dict and 'label' in getattr(self, p['label'])
                assert getattr(self, p['label'])['label'] == p['label']
                setattr(self, p['label'], self._add_parameter_to_optimize(getattr(self, p['label']), overwrite=True))
                # if idx.sum() == 1:
                #     # Scalar parameter
                #     value = stats[key_list[idx][0]][-1]
                #     setattr(self, p['label'], torch.Tensor([value]))
                # else:
                #     # Vector parameter
                #     idx = np.array([p['label'] in k for k in key_list])
                #     if idx.any():
                #         value = build_tensor(stats, key_list[idx])
                #         IPython.embed()
                #         _parameter = getattr(self, p['label'])
                #         _parameter['params'][0].data = value
                #         p = _parameter
                #     else:
                #         print('Warning during loading the checkpoint:')
                #         print('An added optimized parameter must be stored in the state dict to enable loading, `{}` was not.'.format(p['label']))
                #         print('The algorithm will be run using the default parameter, if any specified.')

                        # raise ValueError('An added optimized parameter must be stored in the state dict to enable loading, `{}` was not'.format(p['label']))

    def _store_stats(self, workers_out, unperturbed_out, generation, rank, workers_time):
        # Check existence of required keys. Create if not there.
        ks = filter(lambda k: k != 'seed', workers_out.keys())
        add_dict = {}
        for k in ks: add_dict.update({k + suffix: [] for suffix in ['_min', '_max', '_avg', '_var', '_sum', '_unp']})
        if not set(add_dict.keys()).issubset(set(self.stats.keys())):
            self.stats.update(add_dict)
        # Append data
        self.stats['generations'].append(generation)
        self.stats['walltimes'].append(time.time() - self._training_start_time)
        self.stats['workertimes'].append(workers_time)
        self.stats['unp_rank'].append(rank)
        for i, lr in enumerate(self.lr_scheduler.get_lr()):
            self.stats['lr_' + str(i)].append(lr)
        for k, v in workers_out.items():
            if not k in ['seed']:
                self.stats[k + '_min'].append(np.min(v))
                self.stats[k + '_max'].append(np.max(v))
                self.stats[k + '_avg'].append(np.mean(v))
                self.stats[k + '_var'].append(np.var(v))
                self.stats[k + '_sum'].append(np.sum(v))
                self.stats[k + '_unp'].append(unperturbed_out[k])

    def _dump_stats(self):
        # Store stats as csv file on drive such that self does not grow in size
        # https://stackoverflow.com/questions/23613426/write-dictionary-of-lists-to-a-csv-file
        csvfile_path = os.path.join(self.chkpt_dir, 'stats.csv')
        df = pd.DataFrame(self.stats, index=self.stats['generations'])
        with open(csvfile_path, 'a') as csvfile:
            print_header = os.stat(csvfile_path).st_size == 0
            df.to_csv(csvfile, header=print_header)
        for k in self.stats.keys(): self.stats[k] = []

    def save_checkpoint(self, best_model_stdct=None, best_optimizer_stdct=None, best_algorithm_stdct=None):
        """
        Save a checkpoint of the algorithm.
        
        The checkpoint consists of `self.model` and `self.optimizer` in the latest and best versions along with 
        statistics in the `self.stats` dictionary.

        Args:
            best_model_stdct (dict, optional): Defaults to None. State dictionary of the checkpoint's best model
            best_optimizer_stdct (dict, optional): Defaults to None. State dictionary of the associated optimizer
            best_algorithm_stdct (dict, optional): Defaults to None. State dictionary of the associated algorithm
        """
        if self.chkpt_dir is None:
            return
        # Save stats
        self._dump_stats()
        # Save latest model and optimizer state
        torch.save(self.state_dict(exclude=True), os.path.join(self.chkpt_dir, 'state-dict-algorithm.pkl'))
        torch.save(self.model.state_dict(), os.path.join(self.chkpt_dir, 'state-dict-model.pkl'))
        torch.save(self.optimizer.state_dict(), os.path.join(self.chkpt_dir, 'state-dict-optimizer.pkl'))
        # Save best model
        if best_model_stdct is not None:
            torch.save(self.state_dict(exclude=True), os.path.join(self.chkpt_dir, 'state-dict-best-algorithm.pkl'))
            torch.save(self.model.state_dict(), os.path.join(self.chkpt_dir, 'state-dict-best-model.pkl'))
            torch.save(self.optimizer.state_dict(), os.path.join(self.chkpt_dir, 'state-dict-best-optimizer.pkl'))
        if not self.silent:
            print(' | checkpoint', end='')
        # Currently, learning rate scheduler has no state_dict and cannot be saved. 
        # It can however be restored by setting lr_scheduler.last_epoch = last generation index since
        # this is the only property that has any effect on its functioning.
        # torch.save(lr_scheduler.state_dict(), os.path.join(self.chkpt_dir, 'state-dict-lr-scheduler.pkl'))

    def state_dict(self, exclude=False):
        """Get the state dictionary of the algorithm.

        Args:
            exclude (bool, optional): Defaults to False. Exlude the attributes defined in self.exlude_from_state_dict

        Returns:
            dict: The state dictionary
        """
        if exclude:
            algorithm_state_dict = self.state_dict().copy()
            for k in self.exclude_from_state_dict:
                algorithm_state_dict.pop(k, None)
            return algorithm_state_dict
        else:
            return vars(self)

    def load_state_dict(self, state_dict):
        """Updates the Algorithm state to that of the given state dictionary.

        If the given state dictionary has keys that are not in the specific Algoritm object or
        if the Algorithm object has attributes that are not in the state dictionary, an error is raised
        
        Args:
            state_dict (dict): Dictionary of the attributes of the Algorithm object.
        """
        # Certain attributes should be missing while others must be present
        assert (set(vars(self)) - set(state_dict)) == self.exclude_from_state_dict, 'The loaded state_dict does not correspond to the chosen algorithm'
        for k, v in state_dict.items():
            if k in self.__dict__.keys():
                self.__dict__[k] = v
        if not self.silent:
            print("\n" + self.__class__.__name__ + ' algorithm restored from state dict.')


class EvolutionaryStrategy(Algorithm):
    def __init__(self, model, env, optimizer, lr_scheduler, eval_fun, perturbations, batch_size, max_generations, safe_mutation, no_antithetic, **kwargs):
        super(EvolutionaryStrategy, self).__init__(model, env, optimizer, lr_scheduler, eval_fun, perturbations, batch_size, max_generations, safe_mutation, no_antithetic, **kwargs)

    def _eval_wrap(self, seed, **kwargs):
        """Get a perturbed model and a copy of the environment and evaluate the perturbation.
        """
        model = self.perturb_model(seed)
        # env = copy.deepcopy(self.env)
        return self.eval_fun(model, self.env, seed, **kwargs)

    def train(self):
        # Initialize
        # TODO Maybe possible to move to abstract class either in train method or as distinct methods (e.g. init_train)
        self.print_init()
        self.model.train()
        # Initialize variables dependent on restoring
        if not self._training_start_time:
            # If nothing stored then start from scratch
            start_generation = 0
            max_unperturbed_return = -1e8
            self._training_start_time = time.time()
        else:
            # If something stored then start from last iteration
            start_generation = self.lr_scheduler.last_epoch + 1
            max_unperturbed_return = self._max_unp_return
        # Initialize variables independent of state
        if self.workers > 1: pool = mp.Pool(processes=self.workers)
        if self.track_parallel and "DISPLAY" in os.environ: pb = PoolProgress(pool, update_interval=.5, keep_after_done=False, title='Evaluating perturbations')
        best_algorithm_stdct = None
        best_model_stdct = None
        best_optimizer_stdct = None
        last_checkpoint_time = time.time()
        chunksize = self.perturbations // (10 * self.workers) + int(self.perturbations // (10 * self.workers) == 0)
        eval_kwargs = {'max_episode_length': self.batch_size, 'chunksize': chunksize}
        # eval_kwargs_unp = {'max_episode_length': self.batch_size, 'collect_inputs': hasattr(self.env, 'env'), 'chunksize': chunksize}
        eval_kwargs_unp = {'max_episode_length': self.batch_size, 'collect_inputs': hasattr(self.env, 'env') * self.batch_size, 'chunksize': chunksize}

        if hasattr(self.env, 'env'):
            # unperturbed_out = pool.apply(self.eval_fun, (self.model, self.env, 42), eval_kwargs_unp)
            unperturbed_out = self.eval_fun(self.model, self.env, 42, **eval_kwargs_unp)
            self.sens_inputs = torch.from_numpy(unperturbed_out['inputs'])

        # Start training loop
        for n_generation in range(start_generation, self.max_generations):
            # Compute parent model weight-output sensitivities
            self.compute_sensitivities()

            # Generate random seeds
            seeds = torch.LongTensor(int(self.perturbations/((not self.no_antithetic) + 1))).random_()
            if not self.no_antithetic: seeds = torch.cat([seeds, -seeds])
            assert len(seeds) == self.perturbations, 'Number of created seeds is not equal to wanted perturbations'
            
            # Evaluate perturbations
            workers_start_time = time.time()
            if self.workers > 1:
                # Execute all perturbations on the pool of processes
                workers_out = pool.map_async(partial(self._eval_wrap, **eval_kwargs), seeds)
                unperturbed_out = pool.apply_async(self.eval_fun, (self.model, self.env, 42), eval_kwargs_unp)
                if self.track_parallel and "DISPLAY" in os.environ: pb.track(workers_out)
                workers_out = workers_out.get(timeout=3600)
                unperturbed_out = unperturbed_out.get(timeout=3600)
            else:
                # Execute sequentially
                unperturbed_out = self.eval_fun(self.model, self.env, 42, **eval_kwargs_unp)
                workers_out = []
                for s in seeds:
                    workers_out.append(self._eval_wrap(s, **eval_kwargs))
            workers_time = time.time() - workers_start_time
            assert 'return' in unperturbed_out.keys() and 'seed' in unperturbed_out.keys(), "The `eval_fun` must give a return and repass the used seed"
            if hasattr(self.env, 'env'):
                self.sens_inputs = torch.from_numpy(unperturbed_out['inputs'])
            # Invert output from list of dicts to dict of lists
            workers_out = dict(zip(workers_out[0], zip(*[d.values() for d in workers_out])))
            # Recast all outputs as np.ndarrays except the seeds
            for k, v in filter(lambda i: i[0] != 'seed', workers_out.items()): workers_out[k] = np.array(v)
            
            # Shaping, rank, compute gradients, update parameters and learning rate
            rank = self.unperturbed_rank(workers_out['return'], unperturbed_out['return'])
            shaped_returns = self.fitness_shaping(workers_out['return'])
            self.compute_gradients(shaped_returns, workers_out['seed'])
            self.model.cpu()  # TODO Find out why the optimizer requires model on CPU even with args.cuda = True
            self.optimizer.step()
            if type(self.lr_scheduler) == torch.optim.lr_scheduler.ReduceLROnPlateau:
                self.lr_scheduler.step(unperturbed_out['return'])
            else:
                self.lr_scheduler.step()

            # Keep track of best model
            # TODO bm, bo, ba, mur = self.update_best(unperturbed_out['return'], mur)
            # TODO Maybe not evaluate unperturbed model every iteration
            if unperturbed_out['return'] >= max_unperturbed_return:
                best_model_stdct = self.model.state_dict()
                best_optimizer_stdct = self.optimizer.state_dict()
                best_algorithm_stdct = self.state_dict(exclude=True)
                max_unperturbed_return = unperturbed_out['return']

            # Print and checkpoint
            self._store_stats(workers_out, unperturbed_out, n_generation, rank, workers_time)
            self.print_iter()
            if last_checkpoint_time < time.time() - self.chkpt_int:
                self.save_checkpoint(best_model_stdct, best_optimizer_stdct, best_algorithm_stdct)
                plot_stats(os.path.join(self.chkpt_dir, 'stats.csv'), self.chkpt_dir)
                last_checkpoint_time = time.time()
        
        # End training
        self.save_checkpoint(best_model_stdct, best_optimizer_stdct, best_algorithm_stdct)
        if self.workers > 1:
            pool.close()
            pool.join()
        print("\nTraining done\n\n")

        # # For python multiprocessing: Using apply_async and eval_fun
        # t = time.time()
        # unperturbed_out = pool.apply_async(self.eval_fun, (self.model, self.env, 42), eval_kwargs_unp)
        # workers_out = []
        # for i in range(self.perturbations):
        #     model = self.perturb_model(s)
        #     workers_out.append(pool.apply_async(self.eval_fun, (model, self.env, seeds[i]), eval_kwargs))
        # for w in workers_out:
        #     w.get(600)
        # unperturbed_out.get(600)
        # print(time.time() - t)
        # # 14.81, 18.6

        # # For python multiprocessing: Using starmap and eval_fun
        # t = time.time()
        # unperturbed_out = pool.apply_async(self.eval_fun, (self.model, self.env, 42), eval_kwargs_unp)
        # inputs = []
        # models = []
        # for s in seeds:
        #     model = self.perturb_model(s)
        #     # models.append(model)
        #     inputs.append((model, self.env, s))
        # workers_out = pool.starmap_async(self.eval_fun, inputs)
        # workers_out.get(600)
        # unperturbed_out.get(600)
        # print(time.time() - t)
        # # 34.24, 33.54, 38.23, 35.39

        # # For python multiprocessing: map_async using eval_wrap (original)
        # t = time.time()
        # workers_out = pool.map_async(partial(self._eval_wrap, **eval_kwargs), seeds)
        # unperturbed_out = pool.apply_async(self.eval_fun, (self.model, self.env, 42), eval_kwargs_unp)
        # workers_out.get(600)
        # unperturbed_out.get(600)
        # print(time.time() - t)
        # 17.19, 14.69, 13.86, 15.26, 18.71

        # # Sequential
        # t = time.time()
        # unperturbed_out = self.eval_fun(self.model, self.env, 42, **eval_kwargs_unp)
        # workers_out = []
        # for s in seeds:
        #     workers_out.append(self._eval_wrap(s, **eval_kwargs))
        # print(time.time() - t)
        # # 25.36, 26.03, 32.63, 25.48

        # For Pathos
        # workers_out = pool.amap(self._eval_wrap, seeds, **eval_kwargs)
        # unperturbed_out = pool.amap(self.eval_fun, (self.model, self.env, 42, eval_kwargs_unp))


class ES(EvolutionaryStrategy):
    """Simple regular gradient evolution strategy based on an isotropic Gaussian search distribution.

    The ES algorithm can be derived in the framework of Variational Optimization (VO).
    """
    def __init__(self, model, env, optimizer, lr_scheduler, eval_fun, perturbations, batch_size, max_generations, safe_mutation, no_antithetic, sigma, optimize_sigma=None, cov_lr=None, **kwargs):
        super(ES, self).__init__(model, env, optimizer, lr_scheduler, eval_fun, perturbations, batch_size, max_generations, safe_mutation, no_antithetic, **kwargs)
        self.sigma = sigma
        self.optimize_sigma = optimize_sigma
        self.stats['sigma'] = []
        self.stats['beta'] = []
        # Add beta to optimizer and lr_scheduler
        if self.optimize_sigma is not None:
            beta_val = self._sigma2beta(sigma)
            lr = cov_lr if cov_lr else self.lr_scheduler.get_lr()[0]
            beta_par = {'label': '_beta', 'params': Variable(torch.Tensor([beta_val]), requires_grad=True),
                        'lr': lr, 'weight_decay': 0, 'momentum': 0.9}
            self._beta = self._add_parameter_to_optimize(beta_par)
        # Add learning rates to stats
        for i, _ in enumerate(self.lr_scheduler.get_lr()):
            self.stats['lr_' + str(i)] = []

    def _store_stats(self, *args):
        super(ES, self)._store_stats(*args)
        self.stats['sigma'].append(self.sigma)
        self.stats['beta'].append(self.beta.data[0])

    @property
    def beta(self):
        if self.optimize_sigma:
            assert type(self._beta) is dict and 'params' in self._beta
            beta = self._beta['params'][0]
            self.sigma = self._beta2sigma(beta.data[0])
        else:
            beta = self._sigma2beta(self.sigma)
        return beta
    
    @beta.setter
    def beta(self, beta):
        if self.optimize_sigma:
            self._beta['params'][0].data = beta
        else:
            self._beta = beta
    
    @staticmethod
    def _sigma2beta(sigma):
        return np.log(sigma**2)

    @staticmethod
    def _beta2sigma(beta):
        return np.sqrt(np.exp(beta))
        # return np.sqrt(np.exp(beta)) if type(beta) is np.float64 else (beta.exp()).sqrt().data.numpy()[0]

    def perturb_model(self, seed):
        """Perturbs the main model.
        """
        # Get model class and instantiate new models as copies of parent
        model_class = type(self.model)
        perturbed_model = model_class(self.env.observation_space, self.env.action_space) if hasattr(self.env, 'observation_space') else model_class()
        perturbed_model.load_state_dict(self.model.state_dict())
        perturbed_model.zero_grad()
        # Handle antithetic sampling
        sign = np.sign(seed)
        # Set seed and permute by isotropic Gaussian noise
        torch.manual_seed(abs(seed))
        torch.cuda.manual_seed(abs(seed))
        for pp, sens in zip_longest(perturbed_model.parameters(), self.sensitivities):
            eps = self.get_perturbation(pp.size(), sensitivities=sens)
            pp.data += sign * self.sigma * eps
            if np.isnan(pp.data).any():
                print(pp.data)
                print(sens)
                print(eps)
            assert not np.isnan(pp.data).any()
            assert not np.isinf(pp.data).any()
        return perturbed_model

    def weight_gradient(self, retrn, eps):
        return 1 / (self.perturbations * self.sigma) * (retrn * eps)

    def beta_gradient(self, retrn, eps):
        return 1 / (2 * self.perturbations) * retrn * (eps.pow(2).sum() - eps.numel())

    def compute_gradients(self, returns, seeds):
        """Computes the gradients of the weights of the model wrt. to the return. 
        
        The gradients will point in the direction of change in the weights resulting in a
        decrease in the return.
        """
        # CUDA
        if self.cuda:
            self.model.cuda()

        ## Indpendent parameter groups sampling
        # Preallocate list with gradients
        weight_gradients = []
        beta_gradient = torch.zeros(1)
        for param in self.model.parameters():
            weight_gradients.append(torch.zeros(param.data.size()))

        # Compute gradients
        for i, retrn in enumerate(returns):
            # Set random seed, get antithetic multiplier and return
            sign = np.sign(seeds[i])
            torch.manual_seed(abs(seeds[i]))
            torch.cuda.manual_seed(abs(seeds[i]))
            for layer, param in enumerate(self.model.parameters()):
                eps = self.get_perturbation(param.size(), sensitivities=self.sensitivities[layer], cuda=self.cuda)
                weight_gradients[layer] += self.weight_gradient(sign * retrn, eps)
                if self.optimize_sigma:
                    beta_gradient += self.beta_gradient(retrn, eps)

        # Set gradients
        self.optimizer.zero_grad()
        for layer, param in enumerate(self.model.parameters()):
            param.grad.data = - weight_gradients[layer]
            assert not np.isnan(param.grad.data).any()
            assert not np.isinf(param.grad.data).any()
        if self.optimize_sigma:
            self.beta.grad = - Variable(beta_gradient, requires_grad=True)
            assert not np.isnan(self.beta.grad.data).any()
            assert not np.isinf(self.beta.grad.data).any()

    def print_init(self):
        super(ES, self).print_init()
        s = "Sigma                 {:5.4f}\n".format(self.sigma)
        s += "Optimizing sigma      {:s}\n".format(str(self.optimize_sigma))
        if self.chkpt_dir is not None:
            with open(os.path.join(self.chkpt_dir, 'init.log'), 'a') as f:
                f.write(s + "\n\n")
        s += "\n=================== Running ===================\n"
        print(s)

    def print_iter(self):
        super(ES, self).print_iter()
        s = " | Sig {:5.4f}".format(self.stats['sigma'][-1])
        print(s, end='')


class NES(ES):
    """Simple natural gradient evolution strategy based on an isotropic Gaussian search distribution
    """
    def __init__(self, model, env, optimizer, lr_scheduler, eval_fun, perturbations, batch_size, max_generations, safe_mutation, no_antithetic, sigma, optimize_sigma=None, cov_lr=None, **kwargs):
        super(NES, self).__init__(model, env, optimizer, lr_scheduler, eval_fun, perturbations, batch_size, max_generations, safe_mutation, no_antithetic, sigma, optimize_sigma=optimize_sigma, cov_lr=cov_lr, **kwargs)
        # Scale to enable use of same learning rates as for ES
        self._weight_update_scale = 1/self.sigma**2
        # self._beta_update_scale = 1/(2*self.sigma**4)

    def weight_gradient(self, retrn, eps):
        # Equal to sigma^2 * [regular gradient] (1 / (self.perturbations * self.sigma) * (retrn * eps))
        # To rescale gradients to same size as for ES, we multiply by a factor of 1/sigma_0**2 where
        # sigma_0 is the initial sigma value
        return self._weight_update_scale * (self.sigma / self.perturbations) * (retrn * eps)

    def beta_gradient(self, retrn, eps):
        # Equal to 1/2 * [regular gradient] (1 / (2 * self.perturbations) * retrn * (eps.pow(2).sum() - eps.numel()))
        # Since the natural gradient is independent of beta, there is no scaling required.
        return retrn * (eps.pow(2).sum() - eps.numel()) / (2 * self.perturbations)


class sES(EvolutionaryStrategy):
    """Simple Seperable Evolution Strategy

    An algorithm based on an Evolution Strategy (ES) using a Gaussian search distribution.
    The ES algorithm can be derived in the framework of Variational Optimization (VO).
    """

    def __init__(self, model, env, optimizer, lr_scheduler, eval_fun, perturbations, batch_size, max_generations, safe_mutation, no_antithetic, sigma, optimize_sigma=None, cov_lr=None, **kwargs):
        super(sES, self).__init__(model, env, optimizer, lr_scheduler, eval_fun, perturbations, batch_size, max_generations, safe_mutation, no_antithetic, **kwargs)
        assert optimize_sigma in [None, 'single', 'per-layer', 'per-weight'], "Expected `self.optimize_sigma` to be one of [None, 'single', 'per-layer', 'per-weight'] but was {}".format(self.optimize_sigma)
        self.optimize_sigma = optimize_sigma
        # Add sigma to stats
        if self.optimize_sigma == 'per-weight':
            sigma_keys = ['sigma_avg', 'sigma_min', 'sigma_max', 'sigma_med', 'sigma_std']
            beta_keys = ['beta_avg', 'beta_min', 'beta_max', 'beta_med', 'beta_std']
        elif self.optimize_sigma == 'per-layer':
            n = self.model.count_tensors(only_trainable=True)
            sigma_keys = ['sigma_' + str(i) for i in range(n)]
            beta_keys = ['beta_' + str(i) for i in range(n)]
        else:
            sigma_keys = ['sigma']
            beta_keys = ['beta']
        for sk, bk in zip(sigma_keys, beta_keys):
            self.stats[sk] = []
            self.stats[bk] = []
        # Add sigma to optimizer and lr_scheduler
        beta = self._sigma2beta(sigma)
        if self.optimize_sigma is None:
            self._beta = Variable(torch.Tensor([beta]), requires_grad=False)
        else:
            if self.optimize_sigma == 'single':
                assert not isinstance(sigma, (collections.Sequence, np.ndarray, torch.Tensor))
                beta_tensor = torch.Tensor([beta])
            elif self.optimize_sigma == 'per-layer':
                if not isinstance(sigma, (collections.Sequence, np.ndarray, torch.Tensor)):
                    beta_tensor = torch.Tensor([beta] * self.model.count_tensors(only_trainable=True))
                elif isinstance(sigma, (collections.Sequence, np.ndarray)):
                    beta_tensor = torch.Tensor([beta])
                elif isinstance(sigma, torch.Tensor):
                    beta_tensor = beta
            elif self.optimize_sigma == 'per-weight':
                if not isinstance(sigma, (collections.Sequence, np.ndarray, torch.Tensor)):
                    beta_tensor = torch.Tensor([beta] * self.model.count_parameters(only_trainable=True))
                elif isinstance(sigma, (collections.Sequence, np.ndarray)):
                    beta_tensor = torch.Tensor([beta])
                elif isinstance(sigma, torch.Tensor):
                    beta_tensor = beta
            if cov_lr:
                lr = cov_lr
            else:
                lr = self.lr_scheduler.get_lr()[0]
            beta_par = {'label': '_beta', 'params': Variable(beta_tensor, requires_grad=True),
                        'lr': lr, 'weight_decay': 0, 'momentum': 0.9}
            self._beta = self._add_parameter_to_optimize(beta_par)
        self.sigma = self._beta2sigma(self.beta.data)
        # Add learning rates to stats
        for i, _ in enumerate(self.lr_scheduler.get_lr()):
            self.stats['lr_' + str(i)] = []

    def _store_stats(self, *args):
        super(sES, self)._store_stats(*args)
        self.stats['sigma'].append(self.sigma)
        self.stats['beta'].append(self.beta.data[0])

    @property
    def beta(self):
        if self.optimize_sigma:
            assert type(self._beta) is dict and 'params' in self._beta
            beta = self._beta['params'][0]
            self.sigma = self._beta2sigma(beta.data)
        else:
            beta = self._beta
        return beta
    
    @beta.setter
    def beta(self, beta):
        if self.optimize_sigma:
            self._beta['params'][0].data = beta
        else:
            self._beta = beta
    
    @staticmethod
    def _sigma2beta(sigma):
        return np.log(sigma**2)

    @staticmethod
    def _beta2sigma(beta):
        return np.sqrt(np.exp(beta))
    
    def perturb_model(self, seed):
        """Perturbs the main model.
        """
        # Get model class and instantiate new models as copies of parent
        model_class = type(self.model)
        perturbed_model = model_class(self.env.observation_space, self.env.action_space) if hasattr(self.env, 'observation_space') else model_class()
        perturbed_model.load_state_dict(self.model.state_dict())
        perturbed_model.zero_grad()
        # Handle antithetic sampling
        sign = np.sign(seed)
        # Set seed and permute by isotropic Gaussian noise
        torch.manual_seed(abs(seed))
        torch.cuda.manual_seed(abs(seed))
        if self.optimize_sigma in [None, 'single']:
            for p, pp, sens in zip_longest(self.model.parameters(), perturbed_model.parameters(), self.sensitivities):
                eps = self.get_perturbation(p.size(), sensitivities=sens)
                pp.data += sign * self.sigma * eps
        elif self.optimize_sigma == 'per-layer':
            for layer, (p, pp, sens) in enumerate(zip_longest(self.model.parameters(), perturbed_model.parameters(), self.sensitivities)):
                eps = self.get_perturbation(p.size(), sensitivities=sens)
                pp.data += sign * self.sigma[layer] * eps
        elif self.optimize_sigma == 'per-weight':
            i = 0
            for layer, (p, pp, sens) in enumerate(zip_longest(self.model.parameters(), perturbed_model.parameters(), self.sensitivities)):
                j = i + p.numel()
                eps = self.get_perturbation(p.size(), sensitivities=sens)
                pp.data += sign * (self.sigma[i:j] * eps.view(-1)).view(p.size())
                i = j
        # Check numerical values
        for pp in perturbed_model.parameters():
            assert not np.isnan(pp.data).any()
            assert not np.isinf(pp.data).any()
        return perturbed_model

    def compute_gradients(self, returns, seeds):
        """Computes the gradients of the weights of the model wrt. to the return. 
        
        The gradients will point in the direction of change in the weights resulting in a
        decrease in the return.
        """
        # CUDA
        if self.cuda:
            self.model.cuda()
        
        # Preallocate list with gradients
        weight_gradients = []
        for param in self.model.parameters():
            weight_gradients.append(torch.zeros(param.data.size()))
        beta_gradients = torch.zeros(self.beta.size())

        # Compute gradients
        for i, retrn in enumerate(returns):
            # Set random seed, get antithetic multiplier and return
            sign = np.sign(seeds[i])
            torch.manual_seed(abs(seeds[i]))
            torch.cuda.manual_seed(abs(seeds[i]))
            i = 0
            for layer, param in enumerate(self.model.parameters()):
                eps = self.get_perturbation(param.size(), sensitivities=self.sensitivities[layer], cuda=self.cuda)
                if not self.optimize_sigma:
                    weight_gradients[layer] += (1 / (self.perturbations * self.sigma)) * (sign * retrn * eps)
                if self.optimize_sigma == 'single':
                    weight_gradients[layer] += (1 / (self.perturbations * self.sigma)) * (sign * retrn * eps)
                    beta_gradients += 1 / ( 2 * self.perturbations) * retrn * (eps.pow(2).sum() - eps.numel())
                elif self.optimize_sigma == 'per-layer':
                    weight_gradients[layer] += (1 / (self.perturbations * self.sigma[layer])) * (sign * retrn * eps)
                    beta_gradients[layer] += 1 / ( 2 * self.perturbations) * retrn * (eps.pow(2).sum() - eps.numel())
                elif self.optimize_sigma == 'per-weight':
                    j = i + param.numel()
                    weight_gradients[layer] += (1 / (self.perturbations * self.sigma[i:j].view(weight_gradients[layer].size()))) * (sign * retrn * eps)
                    beta_gradients[i:j] += 1 / ( 2 * self.perturbations) * retrn * (eps.view(-1).pow(2) - 1)
                    i = j
  
        # Set gradients
        self.optimizer.zero_grad()
        for layer, param in enumerate(self.model.parameters()):
            param.grad.data = - weight_gradients[layer]
            assert not np.isnan(param.grad.data).any()
            assert not np.isinf(param.grad.data).any()
        if self.optimize_sigma:
            self.beta.grad = - Variable(beta_gradients, requires_grad=True)
            assert not np.isnan(self.beta.grad.data).any()
            assert not np.isinf(self.beta.grad.data).any()

    def print_init(self):
        super(sES, self).print_init()
        s = "Optimizing sigma      {:s}\n".format(str(self.optimize_sigma))
        if self.optimize_sigma in [None, 'single']:
            s += "Sigma                 {:5.4f}\n".format(self.sigma[0])
        elif self.optimize_sigma in ['per-layer', 'per-weight']:
            s += "Sigma mean            {:5.4f}\n".format(self.sigma.mean())
        if self.chkpt_dir is not None:
            with open(os.path.join(self.chkpt_dir, 'init.log'), 'a') as f:
                f.write(s + "\n\n")
        s += "\n=================== Running ===================\n"
        print(s)

    def print_iter(self):
        super(sES, self).print_iter()
        if self.optimize_sigma == 'single':
            s = " | Sig {:5.4f}".format(self.sigma[0])
            print(s, end='')
        elif self.optimize_sigma in ['per-layer', 'per-weight']:
            s = " | Sig {:5.4f}".format(self.sigma.mean())
            print(s, end='')

    def _store_stats(self, *args):
        super(sES, self)._store_stats(*args)
        if not self.optimize_sigma:
            self.stats['sigma'].append(self.sigma.view(-1)[0])
            self.stats['beta'].append(self.beta.data.view(-1)[0])
        elif self.optimize_sigma == 'single':
            self.stats['sigma'].append(self.sigma.view(-1)[0])
            self.stats['beta'].append(self.beta.data.view(-1)[0])
        elif self.optimize_sigma == 'per-layer':
            for i, s in enumerate(self.sigma):
                self.stats['sigma_' + str(i)].append(s)
            for i, b in enumerate(self.beta.data):
                self.stats['beta_' + str(i)].append(b)
        elif self.optimize_sigma == 'per-weight':
            # Compute mean, min, max, median and std
            self.stats['sigma_avg'].append(self.sigma.mean())
            self.stats['sigma_min'].append(self.sigma.min())
            self.stats['sigma_max'].append(self.sigma.max())
            self.stats['sigma_med'].append(self.sigma.median())
            self.stats['sigma_std'].append(self.sigma.std())
            # Avoid the recomputation for beta
            self.stats['beta_avg'].append(self._sigma2beta(self.stats['sigma_avg'][-1]))
            self.stats['beta_min'].append(self._sigma2beta(self.stats['sigma_min'][-1]))
            self.stats['beta_max'].append(self._sigma2beta(self.stats['sigma_max'][-1]))
            self.stats['beta_med'].append(self._sigma2beta(self.stats['sigma_med'][-1]))
            self.stats['beta_std'].append(self.beta.data.std())


class sNES(sES):
    def __init__(self, model, env, optimizer, lr_scheduler, eval_fun, perturbations, batch_size, max_generations, safe_mutation, no_antithetic, sigma, optimize_sigma=None, **kwargs):
        super(sNES, self).__init__(model, env, optimizer, lr_scheduler, eval_fun, perturbations, batch_size, max_generations, safe_mutation, no_antithetic, sigma, optimize_sigma=optimize_sigma, **kwargs)
        # Gradient updates scales to make gradient sizes similar to those using regular gradients
        if self.optimize_sigma == 'single':
            self._weight_update_scale = 1/self.sigma**2
            # self._sigma_update_scale = 1/(2*self.sigma**4)
        else:
            self._weight_update_scale = 1/self.sigma.mean()**2
            # self._sigma_update_scale = 1/(2*self.sigma.mean()**4)

    def compute_gradients(self, returns, seeds):
        """Computes the gradients of the weights of the model wrt. to the return. 
        
        The gradients will point in the direction of change in the weights resulting in a
        decrease in the return.
        """
        # CUDA
        if self.cuda:
            self.model.cuda()
        
        # Preallocate list with gradients
        weight_gradients = []
        for param in self.model.parameters():
            weight_gradients.append(torch.zeros(param.data.size()))
        beta_gradients = torch.zeros(self.beta.size())

        # Compute gradients
        for i, retrn in enumerate(returns):
            # Set random seed, get antithetic multiplier and return
            sign = np.sign(seeds[i])
            torch.manual_seed(abs(seeds[i]))
            torch.cuda.manual_seed(abs(seeds[i]))
            i = 0
            for layer, param in enumerate(self.model.parameters()):
                eps = self.get_perturbation(param.size(), sensitivities=self.sensitivities[layer], cuda=self.cuda)
                if not self.optimize_sigma:
                    weight_gradients[layer] += self._weight_update_scale * self.sigma * (sign * retrn * eps) / self.perturbations
                if self.optimize_sigma == 'single':
                    weight_gradients[layer] += self._weight_update_scale * self.sigma * (sign * retrn * eps) / self.perturbations
                    beta_gradients += retrn * (eps.pow(2).sum() - eps.numel()) / self.perturbations
                elif self.optimize_sigma == 'per-layer':
                    weight_gradients[layer] += self._weight_update_scale * self.sigma[layer] * (sign * retrn * eps) / self.perturbations
                    beta_gradients[layer] += retrn * (eps.pow(2).sum() - eps.numel()) / self.perturbations
                elif self.optimize_sigma == 'per-weight':
                    j = i + param.numel()
                    weight_gradients[layer] += self._weight_update_scale * self.sigma[i:j].view(weight_gradients[layer].size()) * (sign * retrn * eps) / self.perturbations
                    beta_gradients[i:j] += retrn * (eps.view(-1).pow(2) - 1) / self.perturbations
                    i = j

        # Set gradients
        self.optimizer.zero_grad()
        for layer, param in enumerate(self.model.parameters()):
            param.grad.data = - weight_gradients[layer]
            assert not np.isnan(param.grad.data).any()
            assert not np.isinf(param.grad.data).any()
        if self.optimize_sigma:
            self.beta.grad = - Variable(beta_gradients, requires_grad=True)
            assert not np.isnan(self.beta.grad.data).any()
            assert not np.isinf(self.beta.grad.data).any()


class xNES(EvolutionaryStrategy):
    def __init__(self, model, env, optimizer, lr_scheduler, eval_fun, perturbations, batch_size, max_generations, safe_mutation, no_antithetic, sigma, optimize_sigma=None, **kwargs):
        super(xNES, self).__init__(model, env, optimizer, lr_scheduler, eval_fun, perturbations, batch_size, max_generations, safe_mutation, no_antithetic, **kwargs)
        assert optimize_sigma in ['per-layer', 'per-weight']
        self.optimize_sigma = optimize_sigma
        # Form Sigma as matrix if not already
        if type(sigma) in [float, int]:
            if self.optimize_sigma == 'per-layer':
                Sigma = torch.diag(torch.Tensor([sigma] * self.model.count_tensors(only_trainable=True)))
            elif self.optimize_sigma == 'per-weight':
                Sigma = torch.diag(torch.Tensor([sigma] * self.model.count_parameters(only_trainable=True)))
        else:
            assert type(sigma) is torch.Tensor and \
                   (self.optimize_sigma == 'per-layer' and sigma.numel() == self.model.count_tensors(only_trainable=True)) or \
                   (self.optimize_sigma == 'per-weight' and sigma.numel() == self.model.count_parameters(only_trainable=True))
            Sigma = sigma
        # Decouple shape (B) and scale (sigma) information
        # Get dimensionality of search distribution (Compute 'dimensionality' of problem from perturbations np.exp((self.perturbations - 4) / 3))
        self.d = Sigma.size()[0]
        # Cholesky factorize Sigma
        A = torch.potrf(Sigma)
        # Compute scale: Compute the d'th root of the determinant of A
        self.sigma = np.abs(np.linalg.det(A.numpy())) ** (1 / self.d)
        self.sigma = Variable(torch.Tensor([self.sigma]), requires_grad=True)
        # Compute shape: Normalize cholesky factor of initial Sigma by scale
        self.B = Variable(A/self.sigma.data, requires_grad=True)

        # Find maximal number of elements in any model tensor
        m = 0
        for p in self.model.parameters():
            if p.requires_grad:
                m = int(np.max([m, p.numel()]))
        self.model.max_elements_in_a_tensor = m

        self.compute_sensitivities()
        self.perturb_model(42)

    # @staticmethod
    # def get_perturbation(size, sensitivities=None, cuda=False):
    #     """Draws a perturbation tensor of dimension `size` from a standard normal.

    #     If `sensitivities is not None`, then the perturbation is scaled by these.
    #     If sensitivities are given and `sensitivities.size() == size[1:]` then the `size[0]` is 
    #     intepreted as a number of samples each of which is to be scaled by the given sensitivities.
    #     """
    #     if type(size) in [tuple, list]:
    #         size = torch.Size(size)
    #     elif type(size) in [torch.Size, int]:
    #         pass
    #     else:
    #         raise TypeError("Input `size` must be of type `int`, `list`, `tuple` or `torch.Size` but was `{}`".format(type(size).__name__))
    #     if sensitivities is not None and sensitivities.size() == size[1:]:
    #         samples = size[0]
    #     assert sensitivities is None or sensitivities.size() == size or sensitivities.size() == size[1:], "Sensitivities must match size of perturbation"
    #     if cuda:
    #         eps = torch.cuda.FloatTensor(size)
    #     else:
    #         eps = torch.FloatTensor(size)
    #     eps.normal_(mean=0, std=1)
    #     if sensitivities is not None:
    #         if sensitivities.size() == size[1:]:
    #             for s in range(samples):
    #                 eps[s, ...] /= sensitivities      # Scale by sensitivities
    #                 if eps.numel() > 1:
    #                     eps /= eps.std()              # Rescale to unit variance
    #         else:
    #             eps /= sensitivities     # Scale by sensitivities
    #             if eps.numel() > 1:
    #                 eps /= eps.std()     # Rescale to unit variance
    #     return eps

    def perturb_model(self, seed):
        # Get model class and instantiate new models as copies of parent
        model_class = type(self.model)
        perturbed_model = model_class(self.env.observation_space, self.env.action_space) if hasattr(self.env, 'observation_space') else model_class()
        perturbed_model.load_state_dict(self.model.state_dict())
        perturbed_model.zero_grad()
        # Handle antithetic sampling
        sign = np.sign(seed)
        # Set seed and permute by isotropic Gaussian noise
        torch.manual_seed(abs(seed))
        torch.cuda.manual_seed(abs(seed))
        if self.optimize_sigma == 'per-layer':
            # Sample a tensor of perturbations. A perturbation vector for each layer is in each row (column).
            # The challenge is that not all layers have the same number of perturbations - pad with None, zeros? Alternatives?
            # How to do this efficiently?

            # Max number of elements in a parameter
            m = self.model.max_elements_in_a_tensor()
            l = self.model.count_tensors(only_trainable=True)
            # Get perturbations from search distribution
            eps = self.get_perturbation(torch.Size((l, m)))
            eps = sign * self.sigma.data * self.B.data.transpose(0, 1) @ eps
            # Perturb model parameters
            for l, (pp, s) in enumerate(zip(perturbed_model.parameters(), self.sensitivities)):
                # Get needed number of perturbations
                e = eps[l, :pp.numel()].view(pp.size())
                # print(e.mean(), e.std())
                e = self.scale_by_sensitivities(e, s)
                # print(e.mean(), e.std())
                pp.data += e
                assert not np.isnan(pp.data).any()
                assert not np.isinf(pp.data).any()

            # IPython.embed()

            # plt.figure()
            # plt.hist(eps[0:1,:])
            # plt.savefig('dist.pdf')

            # plt.figure()
            # plt.plot(eps[0,:].numpy(), eps[1,:].numpy(), 'o')
            # plt.savefig('scatter.pdf')
            
            # import matplotlib.pyplot as plt
            
            # B = [[1, 0.4], [0.4, 2]]
            # self.B = Variable(torch.Tensor(B))
            # plt.figure()
            # plt.plot(saved[0,:].numpy(), saved[1,:].numpy(), 'o')
            # plt.savefig('scatter.pdf')

            # eps = torch.zeros(torch.Size([l, m]))
            # for l, (p, s) in enumerate(zip(self.model.parameters(), self.sensitivities)):
            #     IPython.embed()
            #     eps[l, :] = self.get_perturbation(sample_size, sensitivities=s).view(-1)
                # rem = m % p.numel()
                # n_samples = int(m/p.numel()) + int(rem != 0)
                # sample_size = torch.Size([n_samples, *p.size()])
                # if rem == 0:
                #     eps[l, :] = self.get_perturbation(sample_size, sensitivities=s).view(-1)
                # else:
                #     eps[l, :-rem] = self.get_perturbation(sample_size, sensitivities=s).view(-1)
            
            # for e in eps: 
                # e = self.

        elif self.optimize_sigma == 'per-weight':
            # Sample a single vector of perturbations (local coordinates)
            sensitivities = self._modelrepr2vec(self.sensitivities)
            n_perturbs = self.model.count_parameters(only_trainable=True)
            eps = self.get_perturbation(torch.Size([n_perturbs]), sensitivities=sensitivities)
            # Compute the task coordinates and reshape into list like model
            eps = sign * self.sigma * self.B.data.transpose(0, 1) * eps
            eps = self._vec2modelrepr(eps)
            # Permute each model parameter
            for pp, e in zip(perturbed_model.parameters(), eps):
                pp.data += e
                assert not np.isnan(p.data).any()
                assert not np.isinf(p.data).any()
        return perturbed_model

    def compute_gradients(self, returns, seeds):
        """Computes the gradients of the weights of the model wrt. to the return. 
        
        The gradients will point in the direction of change in the weights resulting in a
        decrease in the return.
        """
        # CUDA
        if self.cuda:
            self.model.cuda()

        ## Indpendent parameter groups sampling
        # Preallocate list with gradients
        weight_gradients = []
        beta_gradient = 0
        for param in self.model.parameters():
            weight_gradients.append(torch.zeros(param.data.size()))

        # Dependent parameter groups sampling (requires more memory)
        # Preallocate weight gradients as 1xn vector where n is number of parameters in model
        print("xNES compute gradients")
        IPython.embed()
        delta_gradients = torch.zeros(self.model.count_parameters(only_trainable=True))
        M_gradients = torch.zeros(self.d, self.d)
        B_gradients = torch.zeros(self.d, self.d)
        sigma_gradients = torch.zeros(1)
        # Compute gradients
        for i, retrn in enumerate(returns):
            # Set random seed, get antithetic multiplier and return
            sign = np.sign(seeds[i])
            torch.manual_seed(abs(seeds[i]))
            torch.cuda.manual_seed(abs(seeds[i]))

            
            if self.optimize_sigma == 'per-layer':
                # Max number of elements in a parameter
                m = self.model.max_elements_in_a_tensor()
                l = self.model.count_tensors(only_trainable=True)
                # Get perturbations from search distribution
                eps = self.get_perturbation(torch.Size((l, m)))
                # eps = sign * self.sigma.data * self.B.data.transpose(0, 1) @ eps
                # # Perturb model parameters
                # for l, (pp, s) in enumerate(zip(perturbed_model.parameters(), self.sensitivities)):
                #     # Get needed number of perturbations
                #     e = eps[l, :pp.numel()].view(pp.size())
                #     # print(e.mean(), e.std())
                #     e = self.scale_by_sensitivities(e, s)
                
                weight_gradients += 1 / (self.perturbations * self.sigma) * retrn * eps
            beta_gradients += 1 / (self.perturbations * self.beta.exp()) * retrn * (eps.pow(2).sum() - 1)
        weight_gradients = self._vec2modelrepr(weight_gradients, only_trainable=True)

        # sigma_gradients = sigma_gradients.clamp(max=10)
        # sigma_gradients = self.sigma.data * (sigma_gradients.exp() - 1)

        self.sigma.data = self.sigma.data * np.exp(eta_sigma / 2 * sigma_gradients)

        # # Compute gradients
        # for i, retrn in enumerate(returns):
        #     # Set random seed, get antithetic multiplier and return
        #     sign = np.sign(seeds[i])
        #     torch.manual_seed(abs(seeds[i]))
        #     torch.cuda.manual_seed(abs(seeds[i]))
        #     for layer, param in enumerate(self.model.parameters()):
        #         eps = self.get_perturbation_old(param, sensitivity=self.sensitivities[layer], cuda=self.cuda)
        #         weight_gradients[layer] += self.weight_gradient(sign * retrn, eps)
        #         if self.optimize_sigma:
        #             beta_gradient += self.beta_gradient(retrn, eps)

        # Set gradients
        self.optimizer.zero_grad()
        for layer, param in enumerate(self.model.parameters()):
            param.grad.data = - weight_gradients[layer]
            assert not np.isnan(param.grad.data).any()
            assert not np.isinf(param.grad.data).any()
        # TODO: For each parameter group in the optimizer that is not in the model, update the gradient
        if self.optimize_sigma:
            self.beta.grad = - beta_gradient
            assert not np.isnan(self.beta.grad.data).any()
            assert not np.isinf(self.beta.grad.data).any()

        # # Dependent parameter groups sampling (requires more memory)
        # # Preallocate weight gradients as 1xn vector where n is number of parameters in model
        # IPython.embed()
        # weight_gradients = torch.zeros(self.model.count_parameters(only_trainable=True))
        # beta_gradients = torch.zeros(self.beta.size())
        # # Compute gradients
        # for i, retrn in enumerate(returns):
        #     # Set random seed, get antithetic multiplier and return
        #     sign = np.sign(seeds[i])
        #     torch.manual_seed(abs(seeds[i]))
        #     torch.cuda.manual_seed(abs(seeds[i]))
        #     eps = self.get_perturbation_new(sensitivities=self._modelrepr2vec(self.sensitivities, only_trainable=False), cuda=self.cuda)
        #     weight_gradients += 1 / (self.perturbations * self.sigma) * retrn * eps
        #     beta_gradients += 1 / (self.perturbations * self.beta.exp()) * retrn * (eps.pow(2).sum() - 1)
        # weight_gradients = self._vec2modelrepr(weight_gradients, only_trainable=True)


class GA(ES):
    def __init__(self, model, env, optimizer, lr_scheduler, eval_fun, perturbations, batch_size, max_generations, safe_mutation, no_antithetic, **kwargs):
        super(GA, self).__init__(model, env, optimizer, lr_scheduler, eval_fun, perturbations, batch_size, max_generations, safe_mutation, no_antithetic, **kwargs)

    def train(self):
        # Initialize
        # TODO Maybe possible to move to abstract class either in train method or as distinct methods (e.g. init_train)
        self.print_init()
        # Initialize variables dependent on restoring
        if not self._training_start_time:
            # If nothing stored then start from scratch
            start_generation = 0
            max_unperturbed_return = -1e8
            self._training_start_time = time.time()
        else:
            # If something stored then start from last iteration
            start_generation = self.lr_scheduler.last_epoch + 1
            max_unperturbed_return = self._max_unp_return
        # Initialize variables independent of state
        if self.workers > 1: pool = mp.Pool(processes=self.workers)
        if self.track_parallel and "DISPLAY" in os.environ: pb = PoolProgress(pool, update_interval=.5, keep_after_done=False, title='Evaluating perturbations')
        best_algorithm_stdct = None
        best_model_stdct = None
        best_optimizer_stdct = None
        last_checkpoint_time = time.time()
        chunksize = self.perturbations // (10 * self.workers) + int(self.perturbations // (10 * self.workers) == 0)
        eval_kwargs = {'max_episode_length': self.batch_size, 'chunksize': chunksize}
        # eval_kwargs_unp = {'max_episode_length': self.batch_size, 'collect_inputs': hasattr(self.env, 'env'), 'chunksize': chunksize}
        eval_kwargs_unp = {'max_episode_length': self.batch_size, 'collect_inputs': hasattr(self.env, 'env') * self.batch_size, 'chunksize': chunksize}

        if hasattr(self.env, 'env'):
            # unperturbed_out = pool.apply(self.eval_fun, (self.model, self.env, 42), eval_kwargs_unp)
            unperturbed_out = self.eval_fun(self.model, self.env, 42, **eval_kwargs_unp)
            self.sens_inputs = torch.from_numpy(unperturbed_out['inputs'])

        # Start training loop
        for n_generation in range(start_generation, self.max_generations):
            # Compute parent model weight-output sensitivities
            self.compute_sensitivities()

            # Generate random seeds
            seeds = torch.LongTensor(int(self.perturbations/((not self.no_antithetic) + 1))).random_()
            if not self.no_antithetic: seeds = torch.cat([seeds, -seeds])
            assert len(seeds) == self.perturbations, 'Number of created seeds is not equal to wanted perturbations'
            
            workers_start_time = time.time()
            if self.workers > 1:
                # Execute all perturbations on the pool of processes
                workers_out = pool.map_async(partial(self._eval_wrap, **eval_kwargs), seeds)
                unperturbed_out = pool.apply_async(self.eval_fun, (self.model, self.env, 42), eval_kwargs_unp)
                if self.track_parallel and "DISPLAY" in os.environ: pb.track(workers_out)
                workers_out = workers_out.get(timeout=3600)
                unperturbed_out = unperturbed_out.get(timeout=3600)
            else:
                # Execute sequentially
                unperturbed_out = self.eval_fun(self.model, self.env, 42, **eval_kwargs_unp)
                workers_out = []
                for s in seeds:
                    workers_out.append(self._eval_wrap(s, **eval_kwargs))
            workers_time = time.time() - workers_start_time

            assert 'return' in unperturbed_out.keys() and 'seed' in unperturbed_out.keys(), "The `eval_fun` must give a return and repass the used seed"
            if hasattr(self.env, 'env'):
                self.sens_inputs = torch.from_numpy(unperturbed_out['inputs'])
            # Invert output from list of dicts to dict of lists
            workers_out = dict(zip(workers_out[0], zip(*[d.values() for d in workers_out])))
            # Recast all outputs as np.ndarrays except the seeds
            for k, v in filter(lambda i: i[0] != 'seed', workers_out.items()): workers_out[k] = np.array(v)
            
            # Select best model (hillclimber)
            best_idx = np.argmax(workers_out['return'])
            best_seed = workers_out['seed'][best_idx]
            self.model = self.perturb_model(best_seed)
            rank = self.unperturbed_rank(workers_out['return'], unperturbed_out['return'])

            # Keep track of best model
            # TODO bm, bo, ba, mur = self.update_best(unperturbed_out['return'], mur)
            # TODO Maybe not evaluate unperturbed model every iteration
            if unperturbed_out['return'] >= max_unperturbed_return:
                best_model_stdct = self.model.state_dict()
                best_optimizer_stdct = self.optimizer.state_dict()
                best_algorithm_stdct = self.state_dict(exclude=True)
                max_unperturbed_return = unperturbed_out['return']

            # Print and checkpoint
            self._store_stats(workers_out, unperturbed_out, n_generation, rank, workers_time)
            self.print_iter()
            if last_checkpoint_time < time.time() - self.chkpt_int:
                self.save_checkpoint(best_model_stdct, best_optimizer_stdct, best_algorithm_stdct)
                plot_stats(os.path.join(self.chkpt_dir, 'stats.csv'), self.chkpt_dir)
                last_checkpoint_time = time.time()
        
        # End training
        self.save_checkpoint(best_model_stdct, best_optimizer_stdct, best_algorithm_stdct)
        if self.workers > 1:
            pool.close()
            pool.join()
        print("\nTraining done\n\n")



# def generate_seeds_and_models(args, parent_model, self.env):
#     """
#     Returns a seed and 2 perturbed models
#     """
#     np.random.seed()
#     seed = np.random.randint(2**30)
#     two_models = perturb_model(args, parent_model, seed, self.env)
#     return seed, two_models

# if self.optimize_sigma:
#     # print("beta {:5.2f} | bg {:5.1f}".format(args.beta.data.numpy()[0], args.beta.grad.data.numpy()[0]))
#     # print("update_parameters")
#     new_sigma = (0.5*self.beta.exp()).sqrt().data.numpy()[0]
#     # print(" | New sigma {:5.2f}".format(new_sigma), end="")
#     if new_sigma > self.sigma * 1.2:
#         self.sigma = self.sigma * 1.2
#     elif new_sigma < self.sigma * 0.8:
#         self.sigma = self.sigma * 0.8
#     else:
#         self.sigma = new_sigma
#     self.beta.data = torch.Tensor([np.log(2*self.sigma**2)])

            
# # Adjust max length of episodes
# if hasattr(args, 'not_var_ep_len') and not args.not_var_ep_len:
#     args.batch_size = int(5*max(i_observations))


# def perturb_model(self, seed):
#     """Perturbs the main model.

#     Modifies the main model with a pertubation of its parameters,
#     as well as the mirrored perturbation, and returns both perturbed
#     models.
    
#     Args:
#         seed (int): Known random seed. A number between 0 and 2**32.

#     Returns:
#         dict: A dictionary with keys `models` and `seeds` storing exactly that.
#     """
#     # Antithetic or not
#     reps = 1 if self.no_antithetic else 2
#     seeds = [seed] if self.no_antithetic else [seed, -seed]
#     # Get model class and instantiate new models as copies of parent
#     model_class = type(self.model)
#     models = []
#     parameters = [self.model.parameters()]
#     for i in range(reps):
#         this_model = model_class(self.env.observation_space, self.env.action_space) if hasattr(self.env, 'observation_space') else model_class()
#         this_model.load_state_dict(self.model.state_dict())
#         this_model.zero_grad()
#         models.append(this_model)
#         parameters.append(this_model.parameters())
#     parameters = zip(*parameters)
#     # Set seed and permute by isotropic Gaussian noise
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     for pp, p1, *p2 in parameters:
#         eps = self.get_pertubation(pp)
#         p1.data += self.sigma * eps
#         assert not np.isnan(p1.data).any()
#         assert not np.isinf(p1.data).any()
#         if not self.no_antithetic:
#             p2[0].data -= self.sigma * eps
#             assert not np.isnan(p2[0].data).any()
#             assert not np.isinf(p2[0].data).any()
#             assert not (p1.data == p2[0].data).all()
#     return {'models': models, 'seeds': seeds}

def execute_jobs(self, seeds):
    """Start jobs as processes always keeping live processes to a minimum number 
    by continuously ending dead processes
    """
    return_queue = mp.Queue()
    chunksize = self.perturbations // (10 * self.workers) + int(self.perturbations // (10 * self.workers) == 0)
    eval_kwargs = {'max_episode_length': self.batch_size, 'chunksize': chunksize, 'return_queue': return_queue}
    eval_kwargs_unp = {'max_episode_length': self.batch_size, 'collect_inputs': hasattr(self.env, 'env') * self.batch_size, 'chunksize': chunksize, 'return_queue': return_queue}
    processes = []
    outputs = []
    inputs = (self.model, self.env, 42)
    submit_job(self.eval_fun, inputs, eval_kwargs_unp, processes)
    for seed in seeds:
        inputs = (self.perturb_model(seed), self.env, seed)
        submit_job(self.eval_wrap, inputs, eval_kwargs, processes)
        get_output(outputs, processes, return_queue)
        outputs.append(out)

def submit_job(fun, args, kwargs, processes):
    p = mp.Process(target=fun, args=args, kwargs=kwargs)
    p.start()
    processes.append(p)

def get_output(outputs, processes, return_queue):
    # n_done = self.perturbations + 1 - sum([1 for p in processes if p.is_alive()])
    if len(processes) > self.workers * 3:
        pass
    while not return_queue.empty():
        raw_output.append(return_queue.get(False))
    processes = [p for p in processes if p.is_alive()]
    

def start_jobs(self, models, seeds, return_queue):
    processes = []
    while models:
        perturbed_model = models.pop()
        seed = seeds.pop()
        inputs = (perturbed_model, self.env, return_queue, seed)
        try:
            p = mp.Process(target=self.eval_fun, args=inputs, kwargs={'max_episode_length': self.batch_size})
            p.start()
            processes.append(p)
        except (RuntimeError, BlockingIOError) as E:
            IPython.embed()
    assert len(seeds) == 0
    # Evaluate the unperturbed model as well
    inputs = (self.model.cpu(), self.env, return_queue, 'dummy_seed')
    # TODO: Don't collect inputs during run, instead sample at start for sensitivities etc.
    p = mp.Process(target=self.eval_fun, args=inputs, kwargs={'collect_inputs': 1000, 'max_episode_length': self.batch_size})
    p.start()
    processes.append(p)
    return processes, return_queue

def get_job_outputs(self, processes, return_queue):
    raw_output = []
    while processes:
        # Update live processes
        processes = [p for p in processes if p.is_alive()]
        while not return_queue.empty():
            raw_output.append(return_queue.get(False))
    for p in processes:
        p.join()
    return raw_output


# def get_perturbation_old(self, param=None, sensitivity=None, cuda=False):
#     """This method computes a pertubation vector epsilon from a standard normal.

#     It draws perturbations from a standard Gaussian.
#     The pertubation is placed on the GPU if the parameter is there and 
#     `cuda` is `True`. Safe mutations are performed if self.safe_mutation 
#     is not None.
#     """
#     # TODO: This method could also return the perturbations for all model 
#     #       parameters when e.g. sampling from non-isotropic Gaussian. Then, 
#     #       param could default to None. It should still sample form isotropic 
#     #       Gaussian
#     # Sample standard normal distributed pertubation
#     assert (param is None and sensitivity is None) or (param is not None and sensitivity is not None)
#     if param is None:
#         if self.model.is_cuda and cuda:
#             eps = torch.cuda.FloatTensor(self.model.count_parameters(only_trainable=True))
#         else:
#             eps = torch.FloatTensor(self.model.count_parameters(only_trainable=True))
#         eps.normal_(mean=0, std=1)
#         if self.safe_mutation is not None:
#             eps = eps / sensitivity
#             eps = eps / eps.std()
#     else:
#         if param.is_cuda and cuda:
#             eps = torch.cuda.FloatTensor(param.data.size())
#         else:
#             eps = torch.FloatTensor(param.data.size())
#         eps.normal_(mean=0, std=1)
#         # Scale by sensitivities if using safe mutations
#         if self.safe_mutation is not None:
#             assert sensitivity is not None
#             eps = eps / sensitivity       # Scale by sensitivities
#             eps = eps / eps.std()         # Rescale to unit variance
#     return eps


# sES gradients with Wierstra parameterization
# def compute_gradients(self, returns, seeds):
#     """Computes the gradients of the weights of the model wrt. to the return. 
    
#     The gradients will point in the direction of change in the weights resulting in a
#     decrease in the return.
#     """
#     # CUDA
#     if self.cuda:
#         self.model.cuda()

#     # Preallocate list with gradients
#     weight_gradients = []
#     for param in self.model.parameters():
#         weight_gradients.append(torch.zeros(param.data.size()))
#     sigma_gradients = torch.zeros(self.sigma.size())
    
#     # Compute gradients
#     for i, retrn in enumerate(returns):
#         # Set random seed, get antithetic multiplier and return
#         sign = np.sign(seeds[i])
#         torch.manual_seed(abs(seeds[i]))
#         torch.cuda.manual_seed(abs(seeds[i]))
#         i = 0
#         for layer, param in enumerate(self.model.parameters()):
#             eps = self.get_perturbation(param.size(), sensitivities=self.sensitivities[layer], cuda=self.cuda)
#             if not self.optimize_sigma:
#                 weight_gradients[layer] += (1 / (self.perturbations * self.sigma.data)) * (sign * retrn * eps)
#             if self.optimize_sigma == 'single':
#                 weight_gradients[layer] += (1 / (self.perturbations * self.sigma.data)) * (sign * retrn * eps)
#                 sigma_gradients += 1 / ( 2 * self.perturbations) * retrn * (eps.pow(2).sum() - eps.numel())
#             elif self.optimize_sigma == 'per-layer':
#                 weight_gradients[layer] += (1 / (self.perturbations * self.sigma.data[layer])) * (sign * retrn * eps)
#                 # TODO Sigma gradient size is inversely proportional to number of elements in eps
#                 # TODO The more elements in eps, the closer is eps.pow(2).mean() to 1 and the smaller the gradient
#                 # TODO How does eps.pow(2).mean() - 1 scale with elements in eps? Square root?
#                 sigma_gradients[layer] += 1 / ( 2 * self.perturbations) * retrn * (eps.pow(2).sum() - 1)
#             elif self.optimize_sigma == 'per-weight':
#                 j = i + param.numel()
#                 weight_gradients[layer] += (1 / (self.perturbations * self.sigma.data[i:j].view(weight_gradients[layer].size()))) * (sign * retrn * eps)
#                 sigma_gradients[i:j] += 1 / ( 2 * self.perturbations) * retrn * (eps.view(-1).pow(2) - 1)
#                 i = j
#     if self.optimize_sigma:
#         # Clamp gradients before taking exponential transform to avoid inf
#         # IPython.embed()
#         beta = (2 * self.sigma.data.pow(2)).log()
#         beta_gradients = sigma_gradients.clone()
#         beta -= self._sigma['lr'] * beta_gradients
#         self.sigma.data = (0.5 * beta.exp()).sqrt()

#         # sigma_gradients = sigma_gradients.clamp(max=10)
#         # sigma_gradients = self.sigma.data * (sigma_gradients.exp() - 1)
#         # TODO For this parameterization there is a tendency of increasing sigma rather than decreasing
#         # TODO Use regular gradients

#     # Set gradients
#     self.optimizer.zero_grad()
#     for layer, param in enumerate(self.model.parameters()):
#         # param._grad = - Variable(weight_gradients[layer])
#         param.grad.data.set_( - weight_gradients[layer])
#         assert not np.isnan(param.grad.data).any()
#         assert not np.isinf(param.grad.data).any()
#     # if self.optimize_sigma:
#     #     self.sigma.grad = - Variable(sigma_gradients)
#     #     assert not np.isnan(self.sigma.grad.data).any()
#     #     assert not np.isinf(self.sigma.grad.data).any()


#     # ts = time.time()
#     # for i in range(10000):
#     #     for layer, param in enumerate(self.model.parameters()):
#     #         # param._grad = - Variable(weight_gradients[layer])
#     #         param.grad.data.set_( - weight_gradients[layer])
#     #         assert not np.isnan(param.grad.data).any()
#     #         assert not np.isinf(param.grad.data).any()
#     # dur = time.time() - ts
#     # print(dur)
#     # ts = time.time()
#     # for i in range(10000):
#     #     for layer, param in enumerate(self.model.parameters()):
#     #         # param._grad = - Variable(weight_gradients[layer])
#     #         param.grad.data = - weight_gradients[layer]
#     #         assert not np.isnan(param.grad.data).any()
#     #         assert not np.isinf(param.grad.data).any()
#     # dur = time.time() - ts
#     # print(dur)

# Properties for this implementation
        # @property
        # def sigma(self):
        #     if self.optimize_sigma:
        #         assert type(self._sigma) is dict and 'params' in self._sigma
        #         return self._sigma['params'][0]
        #     else:
        #         assert type(self._sigma) is not dict
        #         return self._sigma

        # @sigma.setter
        # def sigma(self, sigma):
        #     if self.optimize_sigma:
        #         assert isinstance(sigma, (collections.Sequence, np.ndarray, torch.Tensor))
        #         if isinstance(sigma, (collections.Sequence, np.ndarray)):
        #             self._sigma['params'][0].data = torch.Tensor(sigma)
        #         else:
        #             self._sigma['params'][0].data = sigma
        #     else:
        #         assert not isinstance(sigma, (collections.Sequence, np.ndarray, torch.Tensor))
        #         self._sigma = sigma