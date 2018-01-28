import datetime
import math
import os
import pickle
import platform
import pprint
import queue
import time
from abc import ABCMeta, abstractmethod

import IPython
import numpy as np

import gym
import torch
import torch.multiprocessing as mp
from torch.autograd import Variable

from .utils import get_lr, plot_stats, save_checkpoint


class Algorithm(object):
    """Abstract class for algorithms

    Attributes:
        model (torch.nn): A pytorch module model
        env (gym): A gym environment
        optimizer (torch.optim.Optimizer): A pytorch optimizer
        lr_scheduler (torch.optim.lr_scheduler): A pytorch learning rate scheduler
        pertubations (int): 
        batch_size (int): 
        max_generations (int): 
        safe_mutation (str): The version of safe mutations to use. Valid options are `ABS`, `SUM` and `SO`
        chktp_dir (str): The directory to use to save/load checkpoints. If not absolute, it will be appended without overlap to the path of this file when executing
        chkpt_int (int): The interval in seconds between checkpoint saves. If chkpt_int<=0, a checkpoint is made at every generation.
        cuda (bool): Boolean to denote whether or not to use CUDA
        silent (bool): Boolean to denote if executing should be silent (no terminal printing)
    """

    __metaclass__ = ABCMeta

    def __init__(self, model, env, optimizer, lr_scheduler, eval_fun, pertubations, batch_size, max_generations, safe_mutation, chkpt_dir, chkpt_int, cuda, silent):
        # Algorithmic attributes
        self.model = model
        self.env = env
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.eval_fun = eval_fun
        self.safe_mutation = safe_mutation
        self.pertubations = pertubations
        self.batch_size = batch_size
        self.max_generations = max_generations
        # Execution attributes
        self.cuda = cuda
        self.silent = silent
        # Checkpoint attributes
        self.chkpt_dir = chkpt_dir
        self.chkpt_int = chkpt_int
        # Attirbutes to exclude from the state dictionary
        self.exclude_from_state_dict = {'env', 'optimizer', 'lr_scheduler', 'model'}
        # Initialize dict for saving statistics
        stat_keys = ['generations', 'episodes', 'observations', 'walltimes',
                    'return_avg', 'return_var', 'return_max', 'return_min',
                    'return_unp', 'unp_rank', 'lr']
        self.stats = {key: [] for key in stat_keys}  # dict.fromkeys(stat_keys) gives None values
        self.stats['do_monitor'] = ['return_unp', 'lr']
        
    @abstractmethod
    def perturb_model(self, random_seed):
        pass

    @abstractmethod
    def compute_gradients(self, *kwargs):
        pass

    @abstractmethod
    def train(self, *kwargs):
        pass

    @staticmethod
    def unperturbed_rank(returns, unperturbed_return):
        """Computes the rank of the unperturbed model among the pertubations.
        
        Args:
            returns (np.array): Returns of evaluated perturbed models.
            unperturbed_return (float): Return of the unperturbed model.
        
        Returns:
            int: Rank of the unperturbed model among the pertubations
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
        n = len(returns)
        sorted_indices = np.argsort(-returns)
        u = np.zeros(n)
        for k in range(n):
            u[sorted_indices[k]] = np.max([0, np.log(n/2+1)-np.log(k+1)])
        return u/np.sum(u)-1/n

    def get_pertubation(self, param):
        """This method computes a pertubation of a given pytorch parameter.
        """
        if param.is_cuda:
            eps = torch.cuda.FloatTensor(param.data.size())
        else:
            eps = torch.FloatTensor(param.data.size())
        eps.normal_(0, 1)
        # eps = torch.from_numpy(np.random.normal(0, 1, param.data.size())).float()
        # if param.is_cuda:
        #     eps = eps.cuda()
        if self.safe_mutation is not None:
            eps = eps/param.grad.data   # Scale by sensitivities
            eps = eps/eps.std()         # Rescale to zero mean unit
        return eps

    def compute_sensitivities(self, inputs, do_normalize=True, do_numerical=True):
        """
        Computes the output-weight sensitivities of the model given a mini-batch
        of inputs.
        Currently implements the SM-G-ABS version of the sensitivities.

        Reference: Safe Mutations for Deep and Recurrent Neural Networks through Output Gradients [2017]
        """
        # Skip if not required
        if self.safe_mutation is None:
            return
        # Convert input to torch Variable
        if type(inputs) is not Variable:
            inputs = Variable(torch.from_numpy(inputs))
        # Forward pass on input batch
        if self.cuda:
            inputs = inputs.cuda()
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
            raise NotImplementedError
        else:
            raise ValueError('The type ''{:s}'' of safe mutations is unrecognized'.format(self.safe_mutation))

        # Normalize
        if do_normalize:
            m = 0
            for pid in range(len(sensitivities)):
                m = sensitivities[pid].max() if sensitivities[pid].max() > m else m
            for pid in range(len(sensitivities)):
                sensitivities[pid] = sensitivities[pid]/m
        
        # Numerical considerations
        if do_numerical:
            for pid in range(len(sensitivities)):
                # Insensitive parameters are unscaled
                sensitivities[pid][sensitivities[pid] < 1e-5] = 1
                # Clip sensitivities at a large constant value
                sensitivities[pid][sensitivities[pid] > 1e5] = 1e5
        
        # Set sensitivities and assert their values
        self.model.zero_grad()
        for pid, param in enumerate(self.model.parameters()):
            param.grad.data = sensitivities[pid].clone()
            assert not np.isnan(param.grad.data).any()
            assert not np.isinf(param.grad.data).any()

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

    def print_init(self):
        """Print the initial message when training is started
        """
        # Get strings to print
        env_name = self.env.spec.id if hasattr(self.env, 'spec') else self.env.dataset.root.split('/')[-1]
        safe_mutation = self.safe_mutation if self.safe_mutation is not None else 'None'
        # Build init string
        s =    "=================== SYSTEM ====================\n"
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
        s += "Environment           {:s}\n".format(env_name)
        s += "Workers               {:d}\n".format(self.pertubations)
        s += "Generations           {:d}\n".format(self.max_generations)
        s += "Batch size            {:<5d}\n".format(self.batch_size)
        s += "Safe mutation         {:s}\n".format(safe_mutation)
        s += "Using CUDA            {:s}\n".format(str(self.cuda))
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
            G = 'G {0:' + str(len(str(self.max_generations))) + 'd}'
            O = 'O {1:' + str(len(str(self.batch_size*self.max_generations))) + 'd}'
            R = 'R {6:' + str(len(str(self.pertubations))) + 'd}'
            s = "\n" + G + " | " + O + " | F {2:6.2f} | A {3:6.2f} | Ma {4:6.2f} | Mi {5:6.2f} | " + R + " | L {7:5.4f}"
            s = s.format(self.stats['generations'][-1], self.stats['observations'][-1], 
                         self.stats['return_unp'][-1], self.stats['return_avg'][-1], 
                         self.stats['return_max'][-1], self.stats['return_min'][-1], 
                         self.stats['unp_rank'][-1], lr)
            print(s, end="")
        except Exception:
            print('Could not print_iter. Some number too large', end="")

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
            algorithm_state_dict = torch.load(os.path.join(chkpt_dir, 'best_algorithm_state_dict.pth'))
            model_state_dict = torch.load(os.path.join(chkpt_dir, 'best_model_state_dict.pth'))
            optimizer_state_dict = torch.load(os.path.join(chkpt_dir, 'best_optimizer_state_dict.pth'))
        else:
            algorithm_state_dict = torch.load(os.path.join(chkpt_dir, 'algorithm_state_dict.pth'))
            model_state_dict = torch.load(os.path.join(chkpt_dir, 'model_state_dict.pth'))
            optimizer_state_dict = torch.load(os.path.join(chkpt_dir, 'optimizer_state_dict.pth'))
        # with open(os.path.join(chkpt_dir, 'stats.pkl'), 'rb') as filename:
        #     self.stats = pickle.load(filename)
        # Load state dicts
        self.load_state_dict(algorithm_state_dict)
        self.model.load_state_dict(model_state_dict)
        self.optimizer.load_state_dict(optimizer_state_dict)
        self.lr_scheduler.last_epoch = self.stats['generations'][-1]

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

        # Save latest model and optimizer state
        torch.save(self.model.state_dict(), os.path.join(self.chkpt_dir, 'model_state_dict.pth'))
        torch.save(self.optimizer.state_dict(), os.path.join(self.chkpt_dir, 'optimizer_state_dict.pth'))
        # Save latest algorithm state without model, otimizer, lr_scheduler and env
        algorithm_state_dict = self.state_dict().copy()
        entries_to_remove = ('model', 'optimizer', 'lr_scheduler', 'env')
        for k in entries_to_remove:
            algorithm_state_dict.pop(k, None)
        torch.save(algorithm_state_dict, os.path.join(self.chkpt_dir, 'algorithm_state_dict.pth'))
        # Save best model
        if best_model_stdct is not None:
            torch.save(best_model_stdct, os.path.join(self.chkpt_dir, 'best_model_state_dict.pth'))
            torch.save(best_optimizer_stdct, os.path.join(self.chkpt_dir, 'best_optimizer_state_dict.pth'))
            # TODO Save best algorithm state
        
        # Currently, learning rate scheduler has no state_dict and cannot be saved. It can be restored
        # by setting lr_scheduler.last_epoch = last generation index.
        # torch.save(lr_scheduler.state_dict(), os.path.join(self.chkpt_dir, 'lr_scheduler_state_dict.pth'))

        # with open(os.path.join(self.chkpt_dir, 'stats.pkl'), 'wb') as filename:
        #     pickle.dump(self.stats, filename, pickle.HIGHEST_PROTOCOL)

    def state_dict(self):
        """Get the state dictionary of the algorithm.
        
        Returns:
            dict: The state dictionary
        """

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
        # Only the env, model, optimizer and lr_scheduler should be missing; and they should be missing
        assert (set(vars(self)) - set(state_dict)) == self.exclude_from_state_dict
        # Assign values to keys
        for k, v in state_dict.items():
            if k in self.__dict__.keys():
                self.__dict__[k] = v
        # Print
        if not self.silent:
            print(self.__class__.__name__ + ' algorith restored from state dict.')
                # s = 'The loaded state_dict does not correspond to the chosen algorithm.\n'
                # s += 'The loaded state_dict is:\n\n'
                # s += pprint.pformat(state_dict)
                # s += '\n\nThe algorithm state dict is:\n\n'
                # s += pprint.pformat(self.state_dict())
                # print(s)
                # raise KeyError()


class NES(Algorithm):
    """Natural Evolution Strategy

    An algorithm based on a Natural Evolution Strategy (NES) using 
    a Gaussian search distribution.
    The NES algorithm can be derived in the framework of Variational
    Optimization (VO).
    """

    def __init__(self, model, env, optimizer, lr_scheduler, eval_fun, pertubations, batch_size, max_generations, safe_mutation, sigma, optimize_sigma=False, beta=None, chkpt_dir=None, chkpt_int=300, cuda=False, silent=False):
        super(NES, self).__init__(model, env, optimizer, lr_scheduler, eval_fun, pertubations, batch_size, max_generations, safe_mutation, chkpt_dir=chkpt_dir, chkpt_int=chkpt_int, cuda=cuda, silent=silent)
        self.sigma = sigma
        self.optimize_sigma = optimize_sigma
        # TODO Maybe dynamically add self.beta parameter as either float based on sigma or 
        #      Variable(torch.Tensor(trsf(sigma))) based on self.optimize_sigma using 
        #      self.optimizer.add_param_group()
        # self.beta = Variable(torch.Tensor([np.log(2*args.sigma**2)]), requires_grad=True) if self.optimize_sigma else 
        self.beta = beta
        self.stats['sigma'] = []
        assert (self.optimize_sigma and self.beta is not None) or (not self.optimize_sigma and self.beta is None)

    # @property
    # def beta(self):
    #     beta_val = self.sigma2beta()
    #     if self.optimize_sigma and 'beta' not in self.optimizer.param_groups:
    #         beta = Variable(torch.Tensor([np.log(2*args.sigma**2)]), requires_grad=True)
    #     return beta

    # @staticmethod
    # def sigma2beta(sigma=None):
    #     conv = lambda sigma: np.log(2*sigma**2)
    #     if sigma is None:
    #         return conv(self.sigma)
    #     else:
    #         return conv(sigma)

    # @staticmethod
    # def beta2sigma(beta):
    #     if self.optimize_sigma:
    #         sigma = (0.5*args.beta.exp()).sqrt().data.numpy()[0]
    #     else:
    #         sigma = np.sqrt(0.5*np.exp(self.beta))
    #     return sigma

    def perturb_model(self, random_seed):
        """Perturbs the main model.

        Modifies the main model with a pertubation of its parameters,
        as well as the mirrored perturbation, and returns both perturbed
        models.
        
        Args:
            random_seed (int): Known random seed. A number between 0 and 2**32.

        Returns:
            dict: A dictionary with keys `models` and `seeds` storing exactly that.
        """
        # Get model class and instantiate two new models as copies of parent
        model_class = type(self.model)
        model1 = model_class(self.env.observation_space, self.env.action_space) if hasattr(self.env, 'observation_space') else model_class()
        model2 = model_class(self.env.observation_space, self.env.action_space) if hasattr(self.env, 'observation_space') else model_class()
        model1.load_state_dict(self.model.state_dict())  # This does not load the gradients (doesn't matter here though)
        model2.load_state_dict(self.model.state_dict())
        model1.zero_grad()
        model2.zero_grad()
        # np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        # Permute all weights of each model by isotropic Gaussian noise
        for param1, param2, pp in zip(model1.parameters(), model2.parameters(), self.model.parameters()):
            eps = self.get_pertubation(pp)
            param2.data -= self.sigma * eps
            param1.data += self.sigma * eps
            assert not np.isnan(param1.data).any()
            assert not np.isinf(param1.data).any()
            assert not np.isnan(param2.data).any()
            assert not np.isinf(param2.data).any()
        return {'models': [model1, model2], 'seeds': [random_seed, random_seed]}

    def compute_gradients(self, returns, random_seeds, is_anti_list):
        """Computes the gradients of the weights of the model wrt. to the return. 
        
        The gradients will point in the direction of change in the weights resulting in a 
        decrease in the return.
        """
        # Verify input
        batch_size = len(returns)
        assert batch_size == self.pertubations
        assert len(random_seeds) == batch_size

        # Preallocate list with gradients
        weight_gradients = []
        beta_gradient = 0
        for param in self.model.parameters():
            weight_gradients.append(torch.zeros(param.data.size()))

        # Compute gradients
        for i in range(self.pertubations):
            # Set random seed, get antithetic multiplier and return
            # np.random.seed(random_seeds[i])
            multiplier = -1 if is_anti_list[i] else 1
            retrn = returns[i]
            torch.manual_seed(random_seeds[i])
            torch.cuda.manual_seed(random_seeds[i])
            for layer, param in enumerate(self.model.parameters()):
                eps = self.get_pertubation(param)
                weight_gradients[layer] += 1/(self.pertubations*self.sigma**2) * (retrn * multiplier * eps)
                if self.optimize_sigma:
                    beta_gradient += 1/(self.pertubations * self.beta.exp()) * retrn * (eps.pow(2).sum() - 1)

        # Set gradients
        self.optimizer.zero_grad()
        for layer, param in enumerate(self.model.parameters()):
            param.grad.data = - weight_gradients[layer]
            assert not np.isnan(param.grad.data).any()
            assert not np.isinf(param.grad.data).any()
        if self.optimize_sigma:
            self.beta.grad = - beta_gradient
            assert not np.isnan(self.beta.grad.data).any()

    def train(self):
        # Initialize
        # TODO Maybe possible to move to abstract class either in train method or as distinct methods (e.g. init_train)
        self.print_init()
        if not self.stats['episodes']:
            # If nothing stored then start from scratch
            n_episodes = 0
            n_observations = 0
            start_generation = 0
            max_unperturbed_return = -1e8
            self.stats['start_time'] = time.time()
        else:
            # If something stored then start from last iteration
            n_episodes = self.stats['episodes'][-1]
            n_observations = self.stats['observations'][-1]
            start_generation = self.stats['generations'][-1] + 1
            max_unperturbed_return = np.max(self.stats['return_unp'])
            # self.sigma = stats['sigma'][-1]
            # This should be correctly handled by self.load_state_dict method
        # Initialize return queue for multiprocessing
        return_queue = mp.Queue()
        best_model_stdct = None
        best_optimizer_stdct = None
        last_checkpoint_time = time.time()

        # Evaluate parent model
        self.eval_fun(self.model.cpu(), self.env, return_queue, 'dummy_seed', 'dummy_neg', collect_inputs=True)
        unperturbed_out = return_queue.get()
        # Start training loop
        for n_generation in range(start_generation, self.max_generations):
            # Empty list of processes, seeds and models and return queue
            loop_start_time = time.time()
            processes, seeds, models = [], [], []
            return_queue = mp.Queue()

            # Compute parent model weight-output sensitivities
            if self.cuda:
                self.model.cuda()
            self.compute_sensitivities(unperturbed_out['inputs'])
            
            # Generate a list of perturbed models and their known random seeds
            # TODO: This could be be part of the parallel execution (somehow)
            for j in range(int(self.pertubations/2)):
                random_seed = np.random.randint(2**30)
                out = self.perturb_model(random_seed)
                seeds.extend(out['seeds'])
                models.extend(out['models'])
            assert len(seeds) == len(models)
            # Keep track of which perturbations were positive and negative.
            is_negative = True

            # Add all peturbed models to the queue
            # TODO Move to abstract class as method (e.g. start_jobs(inputs) where inputs is a list of tuples like below)
            workers_start_time = time.time()
            while models:
                perturbed_model = models.pop()
                seed = seeds.pop()
                inputs = (perturbed_model, self.env, return_queue, seed, is_negative)
                # inputs = (perturbed_model, np.abs(seed), return_queue, self.env)
                p = mp.Process(target=self.eval_fun, args=inputs)
                p.start()
                processes.append(p)
                is_negative = not is_negative
            assert len(seeds) == 0
            # Evaluate the unperturbed model as well
            inputs = (self.model, self.env, return_queue, 'dummy_seed', 'dummy_neg')
            p = mp.Process(target=self.eval_fun, args=inputs, kwargs={'collect_inputs': True})
            p.start()
            processes.append(p)
            # Get output from processes until all are terminated and join
            raw_output = []
            while processes:
                # Update live processes
                processes = [p for p in processes if p.is_alive()]
                while not return_queue.empty():
                    raw_output.append(return_queue.get(False))
            for p in processes:
                p.join()
            workers_end_time = time.time()
            
            # Split into parts
            # TODO This is all a bit ugly...
            seeds = [out['seed'] for out in raw_output]
            returns = [out['return'] for out in raw_output]
            is_anti_list = [out['is_anti'] for out in raw_output]
            i_observations = [out['n_observations'] for out in raw_output]
            # Get results of unperturbed model
            unperturbed_index = seeds.index('dummy_seed')
            unperturbed_out = raw_output.pop(unperturbed_index)
            assert unperturbed_out['seed'] == 'dummy_seed'
            # Remove unperturbed results from all results
            seeds.pop(unperturbed_index)
            returns.pop(unperturbed_index)
            is_anti_list.pop(unperturbed_index)
            i_observations.pop(unperturbed_index)
            # Cast to numpy
            returns = np.array(returns)
            
            # Shaping, rank, compute gradients, update parameters and learning rate
            if self.cuda:
                self.model = self.model.cuda()
            rank = self.unperturbed_rank(returns, unperturbed_out['return'])
            shaped_returns = self.fitness_shaping(returns)
            self.compute_gradients(shaped_returns, seeds, is_anti_list)
            if self.cuda:
                self.model = self.model.cpu()
            self.optimizer.step()
            if self.optimize_sigma:
                # print("beta {:5.2f} | bg {:5.1f}".format(args.beta.data.numpy()[0], args.beta.grad.data.numpy()[0]))
                # print("update_parameters")
                new_sigma = (0.5*self.beta.exp()).sqrt().data.numpy()[0]
                # print(" | New sigma {:5.2f}".format(new_sigma), end="")
                if new_sigma > self.sigma * 1.2:
                    self.sigma = self.sigma * 1.2
                elif new_sigma < self.sigma * 0.8:
                    self.sigma = self.sigma * 0.8
                else:
                    self.sigma = new_sigma
                self.beta.data = torch.Tensor([np.log(2*self.sigma**2)])
            if type(self.lr_scheduler) == torch.optim.lr_scheduler.ReduceLROnPlateau:
                # TODO Check that this steps correctly (it steps every patience times and what if returns are negative)
                self.lr_scheduler.step(unperturbed_out['return'])
            else:
                self.lr_scheduler.step()

            # Keep track of best model
            if unperturbed_out['return'] >= max_unperturbed_return:
                best_model_stdct = self.model.state_dict()
                best_optimizer_stdct = self.optimizer.state_dict()
                best_algorithm_stdct = self.state_dict()
                max_unperturbed_return = unperturbed_out['return']

            # Update iter variables
            n_episodes += len(returns)
            n_observations += sum(i_observations)
            # Store statistics
            self.stats['generations'].append(n_generation)
            self.stats['episodes'].append(n_episodes)
            self.stats['observations'].append(n_observations)
            self.stats['walltimes'].append(time.time() - self.stats['start_time'])
            self.stats['return_avg'].append(returns.mean())
            self.stats['return_var'].append(returns.var())
            self.stats['return_max'].append(returns.max())
            self.stats['return_min'].append(returns.min())
            self.stats['return_unp'].append(unperturbed_out['return'])
            self.stats['unp_rank'].append(rank)
            self.stats['sigma'].append(self.sigma)
            self.stats['lr'].append(get_lr(self.optimizer))
            
            # # Adjust max length of episodes
            # if hasattr(args, 'not_var_ep_len') and not args.not_var_ep_len:
            #     args.batch_size = int(5*max(i_observations))

            # Print and checkpoint
            self.print_iter()
            if last_checkpoint_time < time.time() - self.chkpt_int:
                plot_stats(self.stats, self.chkpt_dir)
                self.save_checkpoint(best_model_stdct, best_optimizer_stdct, best_algorithm_stdct)
                last_checkpoint_time = time.time()

    def print_init(self):
        super(NES, self).print_init()
        s =  "Sigma                 {:5.4f}\n".format(self.sigma)
        s += "Optimizing sigma      {:s}\n".format(str(self.optimize_sigma))
        with open(os.path.join(self.chkpt_dir, 'init.log'), 'a') as f:
            f.write(s)
        s += "\n=================== Running ===================\n"
        print(s)

    def print_iter(self):
        super(NES, self).print_iter()
        s = "| Sig {:5.4f}".format(self.stats['sigma'][-1])
        print(s, end='')


class xNES(Algorithm):
    def __init__(self):
        raise NotImplementedError

    def perturb_model(self):
        return NES.perturb_model(self)

# def generate_seeds_and_models(args, parent_model, self.env):
#     """
#     Returns a seed and 2 perturbed models
#     """
#     np.random.seed()
#     random_seed = np.random.randint(2**30)
#     two_models = perturb_model(args, parent_model, random_seed, self.env)
#     return random_seed, two_models
