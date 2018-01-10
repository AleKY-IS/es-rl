from __future__ import absolute_import, division, print_function

import math
import os
import time

import IPython
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.legacy.optim as legacyOptim
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.autograd import Variable


def perturb_model(args, model, random_seed, env):
    """
    Modifies the given model with a pertubation of its parameters,
    as well as the mirrored perturbation, and returns both perturbed
    models.
    """
    # Get model class and instantiate two new models as copies of parent
    model_class = type(model)
    model1 = model_class(env.observation_space, env.action_space) if hasattr(env, 'observation_space') else model_class()
    model2 = model_class(env.observation_space, env.action_space) if hasattr(env, 'observation_space') else model_class()
    model1.load_state_dict(model.state_dict())
    model2.load_state_dict(model.state_dict())
    np.random.seed(random_seed)
    # Permute all weights of each model by isotropic Gaussian noise
    for param1, param2 in zip(model1.es_parameters(), model2.es_parameters()):
        eps = np.random.normal(0, 1, param1.data.size())
        param1.data += torch.from_numpy(args.sigma*eps).float()
        param2.data += torch.from_numpy(args.sigma*-eps).float()

    return [model1, model2]


def generate_seeds_and_models(args, parent_model, env):
    """
    Returns a seed and 2 perturbed models
    """
    np.random.seed()
    random_seed = np.random.randint(2**30)
    two_models = perturb_model(args, parent_model, random_seed, env)
    return random_seed, two_models


def fitness_shaping(returns):
    """
    Performs the fitness rank transformation used for CMA-ES.
    Reference: Natural Evolution Strategies [2014]
    """
    n = len(returns)
    sorted_indices = np.argsort(-np.array(returns))
    u = np.zeros(n)
    for k in range(n):
        u[sorted_indices[k]] = np.max([0, np.log(n/2+1)-np.log(k+1)])
    return u/np.sum(u)-1/n                


def unperturbed_rank(returns, unperturbed_results):
    nth_place = 1
    for r in returns:
        if r > unperturbed_results:
            nth_place += 1
    rank_diag = ('%d out of %d (1 means gradient '
                    'is uninformative)' % (nth_place,
                                            len(returns) + 1))
    return rank_diag, nth_place


def print_init(args):
    if print_every > 0:
        print("----- Evolver parameters ------")
        print("Environment:          %s" % args.env_name)
        print("Population:           %s" % args.model_name)
        print("Workers:              %d" % args.n)
        print("Generations:          %d" % args.max_gradient_updates)
        print("Sigma:                %5.4f" % args.sigma)
        print("Learning rate:        %5.4f\n" % args.lr)
        print("Learning rate decay:  %5.4f\n" % args.lr-decay)
        print("Restore point:        %s\n" % args.restore)
        print("------- Running evolver -------")


def print_iter(args, gen, prtint=1):
    if prtint and gen % prtint == 0:
        print("Generation: {:3d} | Reward {: 4.1f} | Time {:4.2f} seconds".format(gen, test_reward, t))


optimConfig = []
reward_unperturbed = []
reward_average = []
reward_max = []
reward_min = []
reward_var = []
episodes = []


def gradient_update(args, parent_model, returns, random_seeds, is_anti_list):
    # Verify input
    batch_size = len(returns)
    assert batch_size == args.n
    assert len(random_seeds) == batch_size
    
    # Shape returns and get rank of unperturbed model
    shaped_returns = fitness_shaping(returns)



    # For each model, generate the same random numbers as we did
    # before, and update parameters. We apply weight decay once.
    # TODO: Move shared code out of branch
    # TODO: Always use an optimizer:
    #       1. Compute gradients (by some method)
    #       2. Set gradients 
    #               for param in model.parameters():
    #                   param.grad = gradients
    #       3. Optimize with optimizer
    #               optimizer.step()

    # gradient = []
    # for i in range(args.n):
    #     np.random.seed(random_seeds[i])
    #     multiplier = -1 if is_anti_list[i] else 1
    #     reward = shaped_returns[i]
    #     # Compute gradients
    #     for k, v in parent_model.es_params():
    #         # pytorch in general
    #         loss.backward()
    #         # ES strategy
    #         eps = np.random.normal(0, 1, v.size())
    #         gradient.append(torch.from_numpy((args.n*args.sigma) * (reward*multiplier*eps)).float())
    #         # Safe mutations
    #         eps = np.random.normal(0, 1, v.size())
    #         (mini-batch of network outputs).backward() # Compute sensitivities and store them in v.grad.data
    #         gradient.append(eps/v.grad.data) # Scale permutations with sensitivities
    #     
    #     # Set gradients
    #     i = 0
    #     for param in parent_model.parameters():
    #         param.grad = gradient[i]
    #     
    #     # Optimize parameters
    #     optimizer.step()

    if args.useAdam:
        globalGrads = None
        for i in range(args.n):
            np.random.seed(random_seeds[i])
            multiplier = -1 if is_anti_list[i] else 1
            sr = shaped_returns[i]

            localGrads = []
            idx = 0
            for param in parent_model.es_parameters():
                eps = np.random.normal(0, 1, param.data.size())
                grad = torch.from_numpy((args.n*args.sigma) * (sr*multiplier*eps)).float()

                localGrads.append(grad)
                
                if len(optimConfig) == idx:
                    optimConfig.append({'learningRate': args.lr})
                idx = idx + 1
            
            if globalGrads is None:
                globalGrads = localGrads
            else:
                for i in range(len(globalGrads)):
                    globalGrads[i] = torch.add(globalGrads[i], localGrads[i])

        idx = 0
        for param in parent_model.es_parameters():
            r, _ = legacyOptim.adam(lambda x:  (1, -globalGrads[idx]), param.data, optimConfig[idx])
            param.data.copy_(r)
            idx = idx + 1
    else:
        # For each model, generate the same random numbers as we did
        # before, and update parameters. We apply weight decay once.
        for i in range(args.n):
            np.random.seed(random_seeds[i])
            multiplier = -1 if is_anti_list[i] else 1
            sr = shaped_returns[i]
            for k, v in parent_model.es_params():
                eps = np.random.normal(0, 1, v.size())
                grad = torch.from_numpy((args.n*args.sigma) * (sr*multiplier*eps)).float()
                v += args.lr * grad
        args.lr *= args.lr_decay

    return parent_model

# TODO: Examine possibility of reusing pool of workers 
#       - p = Pool(args.n)
#       - for _ in range(args.max_gradient_updates):
#       -   for j in range(int(args.n/2)):
#       -       inputs.append((args, perturbed_model, seed, return_queue, env, is_negative))
#       -   p.imap_unordered(eval_fun, args=inputs) 
def train_loop(args, parent_model, env, eval_fun, chkpt_dir):
    # Print init
    print("Num params in network %d" % parent_model.count_parameters())

    num_eps = 0
    total_num_frames = 0
    for _ in range(args.max_gradient_updates):
        loop_start_time = time.clock()
        # Initialize list of processes, seeds and models and return queue
        processes, seeds, models = [], [], []
        return_queue = mp.Queue()

        # Generate a perturbation and its antithesis
        for j in range(int(args.n/2)):
            random_seed, two_models = generate_seeds_and_models(args, parent_model, env)
            # Add twice because we get two models with the same seed
            seeds.append(random_seed)
            seeds.append(random_seed)
            models.extend(two_models)
        assert len(seeds) == len(models)

        # Keep track of which perturbations were positive and negative.
        # Start with negative true because pop() makes us go backwards.
        is_negative = True
        # Add all peturbed models to the queue
        workers_start_time = time.clock()
        while models:
            perturbed_model = models.pop()
            seed = seeds.pop()
            inputs = (args, perturbed_model, seed, return_queue, env, is_negative)
            p = mp.Process(target=eval_fun, args=inputs)
            p.start()
            processes.append(p)
            is_negative = not is_negative
        assert len(seeds) == 0

        # Evaluate the unperturbed model as well
        p = mp.Process(target=eval_fun, args=(args, parent_model, 'dummy_seed', return_queue, env, 'dummy_neg'))
        p.start()
        processes.append(p)

        # Get results of started processes
        for p in processes:
            p.join()
        workers_end_time = time.clock()
        raw_output = [return_queue.get() for p in processes]
                
        # Get all results
        seeds = [out['seed'] for out in raw_output]
        returns = [out['return'] for out in raw_output]
        is_anti_list = [out['is_anti'] for out in raw_output]
        nsteps = [out['nsteps'] for out in raw_output]

        #IPython.embed()
        # Get results of unperturbed model and remove it from all results
        unperturbed_index = seeds.index('dummy_seed')
        unperturbed_out = raw_output.pop(unperturbed_index)
        assert unperturbed_out['seed'] == 'dummy_seed'
        seeds.pop(unperturbed_index)
        returns.pop(unperturbed_index)
        is_anti_list.pop(unperturbed_index)
        nsteps.pop(unperturbed_index)
        
        #IPython.embed()
        # Update gradient
        num_eps += len(returns)
        parent_model = gradient_update(args, parent_model, returns, seeds, is_anti_list)

        if args.variable_ep_len:
            args.max_episode_length = int(5*max(num_frames))

        # Print to console
        # TODO: Fix saving to include episode number etc
        # TODO: Fix saving to be into file's directory
        # TODO: (Save only the model if it is better than the previous one)
        # TODO: Make checkpoint based on time (every 1 minute e.g.)
        rank_diag, rank = unperturbed_rank(returns, unperturbed_out['return'])
        
        episodes.append(num_eps)
        reward_unperturbed.append(unperturbed_out['return'])
        reward_average.append(np.mean(returns))
        reward_max.append(max(returns))
        reward_min.append(min(returns))
        reward_var.append(np.var(returns))
        

        # Plot
        if num_eps % (40*20) == 0:
            print('Updating plot')
            fig = plt.figure()
            pltUnp, = plt.plot(episodes, reward_unperturbed, label='parent')
            pltAvg, = plt.plot(episodes, reward_average, label='average')
            pltMax, = plt.plot(episodes, reward_max, label='max')
            pltMin, = plt.plot(episodes, reward_min, label='min')
            plt.ylabel('rewards')
            plt.xlabel('episode num')
            plt.legend(handles=[pltUnp, pltAvg, pltMax,pltMin])
            fig.savefig(chkpt_dir+'/progress.pdf')
            plt.close(fig)

            # Add plot of time used

            print('Updated plot')
            

        # Save checkpoint
        # - model.state_dict
        # - optimizer.state_dict
        # - data used in plots and prints
        # - settings for algorithm (sigma)
        torch.save(parent_model.state_dict(), os.path.join(chkpt_dir, 'model_state_dict.pth'))

        loop_end_time = time.clock()
        if not args.silent:
            #IPython.embed()
            print('Episode num: %d\n'
                'Average reward: %f\n'
                'Variance in rewards: %f\n'
                'Max reward: %f\n'
                'Min reward: %f\n'
                'Batch size: %d\n'
                'Max episode length: %d\n'
                'Sigma: %f\n'
                'Learning rate: %f\n'
                'Unperturbed reward: %f\n'
                'Unperturbed rank: %s\n' 
                'Using Adam: %r\n\n' %
                (num_eps, reward_average[-1], reward_var[-1], reward_max[-1],
                reward_min[-1], args.n,
                args.max_episode_length, args.sigma, args.lr,
                unperturbed_out['return'], rank_diag, args.useAdam))
            print("Worker time: {}".format(workers_end_time-workers_start_time))
            print("Loop time: {}".format(loop_end_time-loop_start_time))