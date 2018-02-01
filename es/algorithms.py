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

from context import utils
from utils.plot_utils import plot_stats


class Algorithm(object):
    """Abstract class for variational algorithms

    Attributes:
        model (torch.nn): A pytorch module model
        env (gym): A gym environment
        optimizer (torch.optim.Optimizer): A pytorch optimizer
        lr_scheduler (torch.optim.lr_scheduler): A pytorch learning rate scheduler
        pertubations (int): The number of perturbed models to evaluate
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

    def __init__(self, model, env, optimizer, lr_scheduler, eval_fun, pertubations, batch_size, max_generations, safe_mutation, no_antithetic, chkpt_dir, chkpt_int, cuda, silent):
        # Algorithmic attributes
        self.model = model
        self.env = env
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.eval_fun = eval_fun
        self.safe_mutation = safe_mutation
        self.no_antithetic = no_antithetic
        self.pertubations = pertubations
        self.batch_size = batch_size
        self.max_generations = max_generations
        # Execution attributes
        self.cuda = cuda
        self.silent = silent
        # Checkpoint attributes
        self.chkpt_dir = chkpt_dir
        self.chkpt_int = chkpt_int
        # Attributes to exclude from the state dictionary
        self.exclude_from_state_dict = {'env', 'optimizer', 'lr_scheduler', 'model'}
        # Initialize dict for saving statistics
        stat_keys = ['generations', 'episodes', 'observations', 'walltimes',
                    'return_avg', 'return_var', 'return_max', 'return_min',
                    'return_unp', 'unp_rank', 'lr']
        self.stats = {key: [] for key in stat_keys}  # dict.fromkeys(stat_keys) gives None values
        self.stats['do_monitor'] = ['return_unp', 'lr']

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

    def get_pertubation(self, param, cuda=False):
        """This method computes a pertubation of a given pytorch parameter.
        """
        # Sample standard normal distributed pertubation
        if param.is_cuda:
            eps = torch.cuda.FloatTensor(param.data.size())
        else:
            eps = torch.FloatTensor(param.data.size())
        eps.normal_(0, 1)
        # Scale by sensitivities if using safe mutations
        if self.safe_mutation is not None:
            eps = eps/param.grad.data   # Scale by sensitivities
            eps = eps/eps.std()         # Rescale to unit variance
        if not cuda:
            eps = eps.cpu()
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
        outputs = self.model(inputs)
        batch_size = outputs.data.size()[0]
        n_outputs = outputs.data.size()[1]
        if self.cuda:
            t = torch.cuda.FloatTensor(batch_size, n_outputs).fill_(0)
        else:
            t = torch.zeros(batch_size, n_outputs)

        # Compute sensitivities using specified method
        if self.safe_mutation is None:
            # Skip if not required but activate gradients in model first
            outputs.backward(t)
            return
        elif self.safe_mutation == 'ABS':
            sensitivities = self._compute_sensitivities_abs(outputs, t)
        elif self.safe_mutation == 'SUM':
            sensitivities = self._compute_sensitivities_sum(outputs, t)
        elif self.safe_mutation == 'SO':
            raise NotImplementedError('The second order safe mutation (SM-SO) is not yet implemented')
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
        s += "Pertubations          {:d}\n".format(self.pertubations)
        s += "Generations           {:d}\n".format(self.max_generations)
        s += "Batch size            {:<5d}\n".format(self.batch_size)
        s += "Safe mutation         {:s}\n".format(safe_mutation)
        s += "Antithetic sampling   {:s}\n".format(str(not self.no_antithetic))
        s += "CUDA                  {:s}\n".format(str(self.cuda))
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


class NES(Algorithm):
    """Natural Evolution Strategy

    An algorithm based on a Natural Evolution Strategy (NES) using 
    a Gaussian search distribution.
    The NES algorithm can be derived in the framework of Variational
    Optimization (VO).
    """

    def __init__(self, model, env, optimizer, lr_scheduler, eval_fun, pertubations, batch_size, max_generations, safe_mutation, no_antithetic, sigma, optimize_sigma=False, beta=None, chkpt_dir=None, chkpt_int=300, cuda=False, silent=False):
        super(NES, self).__init__(model, env, optimizer, lr_scheduler, eval_fun, pertubations, batch_size, max_generations, safe_mutation, no_antithetic, chkpt_dir=chkpt_dir, chkpt_int=chkpt_int, cuda=cuda, silent=silent)
        self.sigma = sigma
        self.optimize_sigma = optimize_sigma
        self.stats['sigma'] = []
        self.stats['do_monitor'].append('sigma')
        if self.optimize_sigma:
            # Add beta to optimizer and lr_scheduler
            beta_par = {'label': 'beta', 'params': self.beta, 'lr': self.lr_scheduler.get_lr()[0]/10, 'weight_decay': 0}
            self.add_parameter_to_optimize(beta_par)
        # TODO Maybe dynamically add self.beta parameter as either float based on sigma or 
        #      Variable(torch.Tensor(trsf(sigma))) based on self.optimize_sigma using 
        #      self.optimizer.add_param_group()
        # TODO Also fix that sigma is not optimized after restoring training from checkpoint
        # self.beta = Variable(torch.Tensor([np.log(2*args.sigma**2)]), requires_grad=True) if self.optimize_sigma else 
        # self.beta = beta
        # assert (self.optimize_sigma and self.beta is not None) or (not self.optimize_sigma and self.beta is None)

    @property
    def beta(self):
        beta_val = np.log(2*self.sigma**2)
        beta_group = list(filter(lambda group: group['label'] == 'beta', self.optimizer.param_groups))
        if self.optimize_sigma and not beta_group:
            beta = Variable(torch.Tensor([beta_val]), requires_grad=True)
        elif self.optimize_sigma:
            beta = beta_group[0]['params'][0]
        else:
            beta = np.log(2*self.sigma**2)
        return beta

    @staticmethod
    def sigma2beta(sigma):
        return np.log(2*sigma**2)

    @staticmethod
    def beta2sigma(beta):
        return np.sqrt(0.5*np.exp(beta)) if type(beta) is np.float64 else (0.5*beta.exp()).sqrt().data.numpy()[0]

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
        # Antithetic or not
        reps = 1 if self.no_antithetic else 2
        seeds = [random_seed] if self.no_antithetic else [random_seed, -random_seed]
        # Get model class and instantiate new models as copies of parent
        model_class = type(self.model)
        models = []
        parameters = [self.model.parameters()]
        for i in range(reps):
            this_model = model_class(self.env.observation_space, self.env.action_space) if hasattr(self.env, 'observation_space') else model_class()
            this_model.load_state_dict(self.model.state_dict())
            this_model.zero_grad()
            models.append(this_model)
            parameters.append(this_model.parameters())
        parameters = zip(*parameters)
        # Set seed and permute by isotropic Gaussian noise
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        for pp, p1, *p2  in parameters:
            eps = self.get_pertubation(pp)
            p1.data += self.sigma * eps
            assert not np.isnan(p1.data).any()
            assert not np.isinf(p1.data).any()
            if not self.no_antithetic:
                p2[0].data -= self.sigma * eps
                assert not np.isnan(p2[0].data).any()
                assert not np.isinf(p2[0].data).any()
                assert not (p1.data == p2[0].data).all()
        return {'models': models, 'seeds': seeds}

    def compute_gradients(self, returns, random_seeds):
        """Computes the gradients of the weights of the model wrt. to the return. 
        
        The gradients will point in the direction of change in the weights resulting in a
        decrease in the return.
        """
        # Verify input
        batch_size = len(returns)
        assert batch_size == self.pertubations
        assert len(random_seeds) == batch_size

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
            sign = 1 if random_seeds[i] > 0 else -1
            torch.manual_seed(random_seeds[i])
            torch.cuda.manual_seed(random_seeds[i])
            for layer, param in enumerate(self.model.parameters()):
                eps = self.get_pertubation(param, cuda=self.cuda)
                weight_gradients[layer] += 1/(self.pertubations*self.sigma**2) * (retrn * sign * eps)
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
        # Initialize variables dependent on restoring
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
        # Initialize variables independent of state
        return_queue = mp.Queue()
        best_model_stdct = None
        best_optimizer_stdct = None
        last_checkpoint_time = time.time()

        # Evaluate parent model
        self.eval_fun(self.model.cpu(), self.env, return_queue, 'dummy_seed', collect_inputs=True)
        unperturbed_out = return_queue.get()
        # Start training loop
        for n_generation in range(start_generation, self.max_generations):
            # Empty list of processes, seeds and models and return queue
            loop_start_time = time.time()
            processes, seeds, models = [], [], []
            return_queue = mp.Queue()

            # Compute parent model weight-output sensitivities
            self.compute_sensitivities(unperturbed_out['inputs'])
            
            # Generate a list of perturbed models and their known random seeds
            # TODO: This could be be part of the parallel execution (somehow)
            while len(models) < self.pertubations:
                random_seed = np.random.randint(2**30)
                out = self.perturb_model(random_seed)
                seeds.extend(out['seeds'])
                models.extend(out['models'])
            assert len(seeds) == len(models) and len(models) == self.pertubations

            # Add all peturbed models to the queue
            # TODO Move to abstract class as method (e.g. start_jobs(inputs) where inputs is a list of tuples like below)
            # TODO self.start_jobs(models, seeds, return_queue)
            workers_start_time = time.time()
            while models:
                perturbed_model = models.pop()
                seed = seeds.pop()
                inputs = (perturbed_model, self.env, return_queue, seed)
                p = mp.Process(target=self.eval_fun, args=inputs)
                p.start()
                processes.append(p)
            assert len(seeds) == 0
            # Evaluate the unperturbed model as well
            inputs = (self.model.cpu(), self.env, return_queue, 'dummy_seed')
            p = mp.Process(target=self.eval_fun, args=inputs, kwargs={'collect_inputs': True})
            p.start()
            processes.append(p)

            # Get output from processes until all are terminated and join
            # TODO out = self.get_job_outputs(processes)
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
            # TODO This is all a bit ugly. Can't we do something about this?
            seeds = [out['seed'] for out in raw_output]
            returns = [out['return'] for out in raw_output]
            i_observations = [out['n_observations'] for out in raw_output]
            # Get results of unperturbed model
            unperturbed_index = seeds.index('dummy_seed')
            unperturbed_out = raw_output.pop(unperturbed_index)
            assert unperturbed_out['seed'] == 'dummy_seed'
            # Remove unperturbed results from all results
            seeds.pop(unperturbed_index)
            returns.pop(unperturbed_index)
            i_observations.pop(unperturbed_index)
            # Cast to numpy
            returns = np.array(returns)
            
            # Shaping, rank, compute gradients, update parameters and learning rate
            rank = self.unperturbed_rank(returns, unperturbed_out['return'])
            shaped_returns = self.fitness_shaping(returns)
            self.compute_gradients(shaped_returns, seeds)
            self.model.cpu()  # TODO Find out why the optimizer requires model on CPU even with args.cuda = True
            self.optimizer.step()
            self.sigma = self.beta2sigma(self.beta)
            if type(self.lr_scheduler) == torch.optim.lr_scheduler.ReduceLROnPlateau:
                # TODO Check that this steps correctly (it steps every patience times and what if returns are negative)
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

            # Update iter variables
            n_episodes += len(returns)
            n_observations += sum(i_observations)
            # Store statistics
            # TODO self.store_stats()
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
            self.stats['lr'].append(self.lr_scheduler.get_lr())

            # Print and checkpoint
            self.print_iter()
            if last_checkpoint_time < time.time() - self.chkpt_int:
                plot_stats(self.stats, self.chkpt_dir)
                self.save_checkpoint(best_model_stdct, best_optimizer_stdct, best_algorithm_stdct)
                last_checkpoint_time = time.time()
        
        self.save_checkpoint(best_model_stdct, best_optimizer_stdct, best_algorithm_stdct)

    def print_init(self):
        super(NES, self).print_init()
        s =  "Sigma                 {:5.4f}\n".format(self.sigma)
        s += "Optimizing sigma      {:s}\n".format(str(self.optimize_sigma))
        with open(os.path.join(self.chkpt_dir, 'init.log'), 'a') as f:
            f.write(s + "\n\n")
        s += "\n=================== Running ===================\n"
        print(s)

    def print_iter(self):
        super(NES, self).print_iter()
        s = " | Sig {:5.4f}".format(self.stats['sigma'][-1])
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