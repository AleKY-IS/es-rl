import datetime
import math
import os
import pickle
import platform
import pprint
import queue
import time
from itertools import zip_longest
from abc import ABCMeta, abstractmethod
from functools import partial
import copy
from utils.progress import PoolProgress

import IPython
import numpy as np

import gym
import torch
import torch.multiprocessing as mp
from context import utils
from torch.autograd import Variable
from utils.misc import get_inputs_from_dict
from utils.plotting import plot_stats


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

    def __init__(self, model, env, optimizer, lr_scheduler, eval_fun, perturbations, batch_size, max_generations, safe_mutation, no_antithetic, workers=mp.cpu_count(), chkpt_dir=None, chkpt_int=600, cuda=False, silent=False):
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
        self.workers = workers
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
        # Attributes to exclude from the state dictionary
        self.exclude_from_state_dict = {'env', 'optimizer', 'lr_scheduler', 'model'}
        # Initialize dict for saving statistics
        stat_keys = ['generations', 'walltimes', 'workertimes', 'unp_rank', 'lr']
        self.stats = {key: [] for key in stat_keys}
        self.stats['do_monitor'] = ['return_unp', 'lr']

    @abstractmethod
    def perturb_model(self, seed):
        pass

    @abstractmethod
    def compute_gradients(self, *kwargs):
        pass

    @abstractmethod
    def train(self, *kwargs):
        pass

    def add_parameter_to_optimize(self, parameter_group):
        """Adds a parameter group to be optimized.

        The parameter group can consists of an arbitrary number of parameters that 
        will all get the same learning rate. The group is added to the optimizer and
        the lr_scheduler is reinitialized with the same inputs on the updated optimizer.
        
        Args:
            parameter_group (dict): The parameter group
        """
        self.optimizer.add_param_group(parameter_group)            
        inputs = get_inputs_from_dict(self.lr_scheduler.__class__.__init__, vars(self.lr_scheduler))
        self.lr_scheduler = self.lr_scheduler.__class__(**inputs)

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
        if type(returns) is list:
            returns = np.array(returns)
        n = len(returns)
        sorted_indices = np.argsort(- returns)
        u = np.zeros(n)
        for k in range(n):
            u[sorted_indices[k]] = np.max([0, np.log(n / 2 + 1) - np.log(k + 1)])
        return u / np.sum(u) - 1 / n

    def eval_wrap(self, seed, **kwargs):
        # Get a model and a copy of the environment and evaluate
        model = self.perturb_model(seed)
        env = copy.deepcopy(self.env)
        return self.eval_fun(model, env, seed, **kwargs)

    def get_pertubation(self, param, sensitivity=None, cuda=False):
        """This method computes a pertubation of a given pytorch parameter.

        As of now, it draws perturbations from a standard Gaussian.
        The pertubation is placed on the GPU if the parameter is there and 
        `cuda` is `True`. Safe mutations are performed if self.safe_mutation 
        is not None.
        """
        # TODO: This method could also return the perturbations for all model 
        #       parameters when e.g. sampling from non-isotropic Gaussian. Then, 
        #       param could default to None. It should still sample form isotropic 
        #       Gaussian
        # Sample standard normal distributed pertubation
        if param.is_cuda and cuda:
            eps = torch.cuda.FloatTensor(param.data.size())
        else:
            eps = torch.FloatTensor(param.data.size())
        eps.normal_(mean=0, std=1)
        # Scale by sensitivities if using safe mutations
        if self.safe_mutation is not None:
            assert sensitivity is not None
            eps = eps / sensitivity       # Scale by sensitivities
            eps = eps / eps.std()         # Rescale to unit variance
        return eps

    def compute_sensitivities(self, inputs, do_normalize=True, do_numerical=True):
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
        if type(inputs) is not Variable:
            inputs = Variable(torch.from_numpy(inputs))
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

        # Normalize
        if do_normalize:
            m = 0
            for pid in range(len(sensitivities)):
                m = sensitivities[pid].max() if sensitivities[pid].max() > m else m
            for pid in range(len(sensitivities)):
                sensitivities[pid] = sensitivities[pid] / m
        
        # Numerical considerations
        if do_numerical:
            for pid in range(len(sensitivities)):
                # Insensitive parameters are unscaled
                sensitivities[pid][sensitivities[pid] < 1e-5] = 1
                # Clip sensitivities at a large constant value
                sensitivities[pid][sensitivities[pid] > 1e5] = 1e5
        
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
        s += pprint.pformat(self.model) + "\n"
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
        lr = self.stats['lr'][-1] if type(self.stats['lr'][-1]) is not list else self.stats['lr'][-1][0]
        try:
            # O = 'O {1:' + str(len(str(self.batch_size * self.max_generations * self.perturbations))) + 'd}'
            G = 'G {0:' + str(len(str(self.max_generations))) + 'd}'
            R = 'R {5:' + str(len(str(self.perturbations))) + 'd}'
            s = G + ' | F {1:6.2f} | A {2:6.2f} | Ma {3:6.2f} | Mi {4:6.2f} | ' + R + ' | L {6:5.4f}'
            s = s.format(self.stats['generations'][-1], self.stats['return_unp'][-1],
                         self.stats['return_avg'][-1],  self.stats['return_max'][-1],
                         self.stats['return_min'][-1],  self.stats['unp_rank'][-1], lr)
            if 'accuracy_unp' in self.stats.keys():
                s += ' | C {0:5.1f}%'.format(self.stats['accuracy_unp'][-1]*100)
            print(s, end='')
        except Exception:
            print('Could not print_iter', end='')

    def load_checkpoint(self, chkpt_dir, load_best=False):
        """Loads a saved checkpoint.
        
        Args:
            chkpt_dir (str): Path to the checkpoint directory
            load_best (bool, optional): Defaults to False. Denotes whether or not to load the best model encountered. Otherwise the latest will be loaded.
        
        Raises:
            IOError: If the loading fails, an IOError exception is raised
        """
        # Get state dicts
        if load_best:
            algorithm_state_dict = torch.load(os.path.join(chkpt_dir, 'state-dict-best-algorithm.pkl'))
            model_state_dict = torch.load(os.path.join(chkpt_dir, 'state-dict-best-model.pkl'))
            optimizer_state_dict = torch.load(os.path.join(chkpt_dir, 'state-dict-best-optimizer.pkl'))
        else:
            algorithm_state_dict = torch.load(os.path.join(chkpt_dir, 'state-dict-algorithm.pkl'))
            model_state_dict = torch.load(os.path.join(chkpt_dir, 'state-dict-model.pkl'))
            optimizer_state_dict = torch.load(os.path.join(chkpt_dir, 'state-dict-optimizer.pkl'))
        # Load state dicts
        self.load_state_dict(algorithm_state_dict)
        self.model.load_state_dict(model_state_dict)
        self.optimizer.load_state_dict(optimizer_state_dict)
        self.lr_scheduler.last_epoch = self.stats['generations'][-1]

    def store_stats(self, workers_out, unperturbed_out, generation, rank, workers_time):
        # Check existence of required keys. Create if not there.
        ks = filter(lambda k: k != 'seed', workers_out.keys())
        add_dict = {}
        for k in ks: add_dict.update({k + suffix: [] for suffix in ['_min', '_max', '_avg', '_var', '_sum', '_unp']})
        if not set(add_dict.keys()).issubset(set(self.stats.keys())):
            self.stats.update(add_dict)
        # Append data
        self.stats['generations'].append(generation)
        self.stats['walltimes'].append(time.time() - self.stats['start_time'])
        self.stats['workertimes'].append(workers_time)
        self.stats['unp_rank'].append(rank)
        self.stats['lr'].append(self.lr_scheduler.get_lr())
        for k, v in workers_out.items():
            if not k in ['seed']:
                self.stats[k + '_min'].append(np.min(v))
                self.stats[k + '_max'].append(np.max(v))
                self.stats[k + '_avg'].append(np.mean(v))
                self.stats[k + '_var'].append(np.var(v))
                self.stats[k + '_sum'].append(np.sum(v))
                self.stats[k + '_unp'].append(unperturbed_out[k])

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
        # Currently, learning rate scheduler has no state_dict and cannot be saved. It can be restored
        # by setting lr_scheduler.last_epoch = last generation index.
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
            entries_to_remove = ('model', 'optimizer', 'lr_scheduler', 'env')
            for k in entries_to_remove:
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
        
        Raises:
            KeyError: Raised if the specific Algorithm and the dictionary have any keys that are not in common
        """
        # Certain attributes should be missing while others must be present
        assert (set(vars(self)) - set(state_dict)) == self.exclude_from_state_dict, 'The loaded state_dict does not correspond to the chosen algorithm'
        for k, v in state_dict.items():
            if k in self.__dict__.keys():
                self.__dict__[k] = v
        if not self.silent:
            print(self.__class__.__name__ + ' algorithm restored from state dict.')


class ES(Algorithm):
    """Simple Evolution Strategy

    An algorithm based on an Evolution Strategy (ES) using 
    a Gaussian search distribution.
    The ES algorithm can be derived in the framework of Variational
    Optimization (VO).
    """

    def __init__(self, model, env, optimizer, lr_scheduler, eval_fun, perturbations, batch_size, max_generations, safe_mutation, no_antithetic, sigma, optimize_sigma=False, beta=None, **kwargs):
        super(ES, self).__init__(model, env, optimizer, lr_scheduler, eval_fun, perturbations, batch_size, max_generations, safe_mutation, no_antithetic, **kwargs)
        self.sigma = sigma
        self.optimize_sigma = optimize_sigma
        self.stats['do_monitor'].append('sigma')
        self.stats['sigma'] = []
        if self.optimize_sigma:
            # Add beta to optimizer and lr_scheduler
            beta_val = np.log(2 * self.sigma**2)
            beta_par = {'label': 'beta', 'params': Variable(torch.Tensor([beta_val]), requires_grad=True),
                        'lr': self.lr_scheduler.get_lr()[0] / 10, 'weight_decay': 0}
            self.add_parameter_to_optimize(beta_par)

    def store_stats(self, *args):
        super(ES, self).store_stats(*args)
        self.stats['sigma'].append(self.sigma)

    @property
    def beta(self):
        beta_group = list(filter(lambda group: group['label'] == 'beta', self.optimizer.param_groups))
        if self.optimize_sigma:
            beta = beta_group[0]['params'][0]
        else:
            beta = np.log(2 * self.sigma**2)
        return beta

    @staticmethod
    def sigma2beta(sigma):
        return np.log(2 * sigma**2)

    @staticmethod
    def beta2sigma(beta):
        return np.sqrt(0.5 * np.exp(beta)) if type(beta) is np.float64 else (0.5 * beta.exp()).sqrt().data.numpy()[0]

    def perturb_model(self, seed):
        """Perturbs the main model.

        Modifies the main model with a pertubation of its parameters.
        
        Args:
            seed (int): Known random seed. A number between 0 and 2**32.

        Returns:
            torch.nn.module: The model
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
        for p, pp, sens in zip_longest(self.model.parameters(), perturbed_model.parameters(), self.sensitivities):
            eps = self.get_pertubation(p, sensitivity=sens)
            pp.data += sign * self.sigma * eps
            assert not np.isnan(pp.data).any()
            assert not np.isinf(pp.data).any()
        return perturbed_model

    def weight_gradient_update(self, retrn, eps):
        return 1 / (self.perturbations * self.sigma) * (retrn * eps)

    def beta_gradient_update(self, retrn, eps):
        return 1 / (self.perturbations * self.beta.exp()) * retrn * (eps.pow(2).sum() - 1)

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
        beta_gradient = 0
        for param in self.model.parameters():
            weight_gradients.append(torch.zeros(param.data.size()))

        # Compute gradients
        for i, retrn in enumerate(returns):
            # Set random seed, get antithetic multiplier and return
            sign = np.sign(seeds[i])
            torch.manual_seed(abs(seeds[i]))
            torch.cuda.manual_seed(abs(seeds[i]))
            for layer, param in enumerate(self.model.parameters()):
                eps = self.get_pertubation(param, sensitivity=self.sensitivities[layer], cuda=self.cuda)
                # TODO Create small gradient update method
                # weight_gradients[layer] += 1 / (self.perturbations * self.sigma**2) * (retrn * sign * eps)
                weight_gradients[layer] += self.weight_gradient_update(sign * retrn, eps)
                if self.optimize_sigma:
                    # beta_gradient += 1 / (self.perturbations * self.beta.exp()) * retrn * (eps.pow(2).sum() - 1)
                    beta_gradient += self.beta_gradient_update(retrn, eps)

        # Set gradients
        self.optimizer.zero_grad()
        for layer, param in enumerate(self.model.parameters()):
            # IPython.embed()
            # if not hasattr(param.grad, 'data'):
            #     param.grad.data = torch.zeros(param.size())
            param.grad.data = - weight_gradients[layer]
            assert not np.isnan(param.grad.data).any()
            assert not np.isinf(param.grad.data).any()
        # TODO: For each parameter group in the optimizer that is not in the model, update the gradient
        if self.optimize_sigma:
            self.beta.grad = - beta_gradient
            assert not np.isnan(self.beta.grad.data).any()
            assert not np.isinf(self.beta.grad.data).any()

    def train(self):
        # Initialize
        # TODO Maybe possible to move to abstract class either in train method or as distinct methods (e.g. init_train)
        self.print_init()
        # Initialize variables dependent on restoring
        if not self.stats['generations']:
            # If nothing stored then start from scratch
            start_generation = 0
            max_unperturbed_return = -1e8
            self.stats['start_time'] = time.time()
        else:
            # If something stored then start from last iteration
            # n_episodes = self.stats['episodes'][-1]
            # n_observations = self.stats['observations'][-1]
            start_generation = self.stats['generations'][-1] + 1
            max_unperturbed_return = np.max(self.stats['return_unp'])
        # Initialize variables independent of state
        pool = mp.Pool(processes=self.workers)
        pb = PoolProgress(pool, update_interval=1, keep_after_done=False, title='Evaluating perturbations')
        best_model_stdct = None
        best_optimizer_stdct = None
        last_checkpoint_time = time.time()

        # Evaluate parent model
        # TODO: This can be removed if we sample observations instead 
        unperturbed_out = self.eval_fun(self.model.cpu(), self.env, 'dummy_seed', collect_inputs=1000, max_episode_length=self.batch_size)

        # Start training loop
        for n_generation in range(start_generation, self.max_generations):
            # Compute parent model weight-output sensitivities
            self.compute_sensitivities(unperturbed_out['inputs'])

            # Generate random seeds
            seeds = torch.LongTensor(int(self.perturbations/((not self.no_antithetic) + 1))).random_()
            if not self.no_antithetic: seeds = torch.cat([seeds, -seeds])
            assert len(seeds) == self.perturbations, 'Number of created seeds is not equal to wanted perturbations'

            # Execute all perturbations on the pool of processes
            workers_start_time = time.time()
            kwargs = {'max_episode_length': self.batch_size}
            workers_out = pool.map_async(partial(self.eval_wrap, payload=kwargs), seeds, chunksize=10)
            unperturbed_out = pool.apply_async(self.eval_fun, (self.model, self.env, 'dummy_seed'), {'max_episode_length': self.batch_size, 'collect_inputs': True})
            if "DISPLAY" in os.environ: pb.track(workers_out)
            workers_out = workers_out.get(timeout=600)
            unperturbed_out = unperturbed_out.get(timeout=600)
            workers_time = time.time() - workers_start_time
            assert 'return' in unperturbed_out.keys() and 'seed' in unperturbed_out.keys(), "An evaluation function must give a return and the used seed"
            # Invert output from list of dicts to dict of lists
            workers_out = dict(zip(workers_out[0],zip(*[d.values() for d in workers_out])))
            for k, v in filter(lambda i: i[0] != 'seed', workers_out.items()): workers_out[k] = np.array(v)

            # Shaping, rank, compute gradients, update parameters and learning rate
            rank = self.unperturbed_rank(workers_out['return'], unperturbed_out['return'])
            shaped_returns = self.fitness_shaping(workers_out['return'])
            self.compute_gradients(shaped_returns, workers_out['seed'])
            self.model.cpu()  # TODO Find out why the optimizer requires model on CPU even with args.cuda = True
            self.optimizer.step()
            self.sigma = self.beta2sigma(self.beta)
            # if self.optimizer.__class__ == 'torch.optim.sgd.SGD':
            if type(self.lr_scheduler) == torch.optim.lr_scheduler.ReduceLROnPlateau:
                self.lr_scheduler.step(unperturbed_out['return'])
            else:
                self.lr_scheduler.step()

            # Keep track of best model
            # TODO bm, bo, ba, mur = self.update_best(unperturbed_out['return'], mur)
            if unperturbed_out['return'] >= max_unperturbed_return:
                best_model_stdct = self.model.state_dict()
                best_optimizer_stdct = self.optimizer.state_dict()
                best_algorithm_stdct = self.state_dict(exclude=True)
                max_unperturbed_return = unperturbed_out['return']

            # Print and checkpoint
            self.store_stats(workers_out, unperturbed_out, n_generation, rank, workers_time)
            self.print_iter()
            if last_checkpoint_time < time.time() - self.chkpt_int:
                plot_stats(self.stats, self.chkpt_dir)
                self.save_checkpoint(best_model_stdct, best_optimizer_stdct, best_algorithm_stdct)
                last_checkpoint_time = time.time()
            print()
        
        # End training
        self.save_checkpoint(best_model_stdct, best_optimizer_stdct, best_algorithm_stdct)
        pool.close()
        pool.join()
        print("\nTraining done\n\n")

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
    def __init__(self, model, env, optimizer, lr_scheduler, eval_fun, perturbations, batch_size, max_generations, safe_mutation, no_antithetic, sigma, optimize_sigma=False, beta=None, chkpt_dir=None, chkpt_int=300, cuda=False, silent=False):
        super(NES, self).__init__(model, env, optimizer, lr_scheduler, eval_fun, perturbations, batch_size, max_generations, safe_mutation, no_antithetic, sigma, optimize_sigma=optimize_sigma, beta=beta, chkpt_dir=chkpt_dir, chkpt_int=chkpt_int, cuda=cuda, silent=silent)
        IPython.embed()

    def weight_gradient_update(self, retrn, eps):
        return self.sigma / self.perturbations * (retrn * eps)

    def beta_gradient_update(self, retrn, eps):
        return self.beta.exp() / (2*self.perturbations) * retrn * (eps.pow(2).sum() - 1)








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

# def start_jobs(self, models, seeds, return_queue):
#     processes = []
#     while models:
#         perturbed_model = models.pop()
#         seed = seeds.pop()
#         inputs = (perturbed_model, self.env, return_queue, seed)
#         try:
#             p = mp.Process(target=self.eval_fun, args=inputs, kwargs={'max_episode_length': self.batch_size})
#             p.start()
#             processes.append(p)
#         except (RuntimeError, BlockingIOError) as E:
#             IPython.embed()
#     assert len(seeds) == 0
#     # Evaluate the unperturbed model as well
#     inputs = (self.model.cpu(), self.env, return_queue, 'dummy_seed')
#     # TODO: Don't collect inputs during run, instead sample at start for sensitivities etc.
#     p = mp.Process(target=self.eval_fun, args=inputs, kwargs={'collect_inputs': 1000, 'max_episode_length': self.batch_size})
#     p.start()
#     processes.append(p)
#     return processes, return_queue

# def get_job_outputs(self, processes, return_queue):
#     raw_output = []
#     while processes:
#         # Update live processes
#         processes = [p for p in processes if p.is_alive()]
#         while not return_queue.empty():
#             raw_output.append(return_queue.get(False))
#     for p in processes:
#         p.join()
#     return raw_output