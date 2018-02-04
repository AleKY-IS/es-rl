import time

import IPython
import numpy as np
import scipy.stats as st
from sklearn.metrics import confusion_matrix

import torch
import torch.nn.functional as F
from torch.autograd import Variable


def gym_rollout(model, env, return_queue, random_seed, silent=False, collect_inputs=False, do_cuda=False, max_episode_length=int(1e6), **kwargs):
    """
    Function to do rollouts of a policy defined by `model` in given environment
    """
    # Reset environment
    # print("Work started")
    # model_time = 0
    state = env.reset()
    state = Variable(torch.from_numpy(state).float(), requires_grad=True).unsqueeze(0)
    retrn = 0
    n_observations = 0
    done = False
    if collect_inputs:
        # Collect `collect_inputs` observations
        prealdim = (int(collect_inputs),)
        for d in state.size()[1:]:
            prealdim = prealdim + (d,)
        inputs = torch.zeros(prealdim)
        
    # Rollout
    while not done and n_observations < max_episode_length:
        # Collect states as batch inputs 
        if collect_inputs and collect_inputs > n_observations:
            inputs[n_observations,] = state.data
        # Choose action
        actions = model(state)
        action = actions.max(1)[1].data.numpy()
        # Step
        state, reward, done, _ = env.step(action[0])
        retrn += reward
        n_observations += 1
        # Cast state
        state = Variable(torch.from_numpy(state).float(), requires_grad=True).unsqueeze(0)
    out = {'seed': random_seed, 'return': retrn, 'n_observations': n_observations}
    if collect_inputs:
        if n_observations < collect_inputs:
            inputs = inputs[:n_observations,]
        out['inputs'] = inputs.numpy()
    return_queue.put(out)


def gym_render(model, env, max_episode_length):
    """
    Renders the learned model on the environment for testing.
    """
    try:
        while True:
            # Reset environment
            state = env.reset()
            state = Variable(torch.from_numpy(state).float(), volatile=True).unsqueeze(0)
            this_model_return = 0
            this_model_num_steps = 0
            done = False
            # Rollout
            while not done and this_model_num_steps < max_episode_length:
                # Choose action
                actions = model(state)
                action = actions.max(1)[1].data.numpy()
                # Step
                state, reward, done, _ = env.step(action[0])
                this_model_return += reward
                this_model_num_steps += 1
                # Cast state
                state = Variable(torch.from_numpy(state).float(), volatile=True).unsqueeze(0)
                env.render()
            print('Reward: %f' % this_model_return)
    except KeyboardInterrupt:
        print("\nEnded test session by keyboard interrupt")


def gym_test(model, env, max_episode_length, n_episodes, **kwargs):
    """
    Tests the learned model on the environment.
    """
    returns = [0]*n_episodes
    for i_episode in range(n_episodes):
        print('Episode {:d}/{:d}'.format(i_episode, n_episodes))
        # Reset environment
        state = env.reset()
        state = Variable(torch.from_numpy(state).float(), volatile=True).unsqueeze(0)
        this_model_num_steps = 0
        done = False
        # Rollout
        while not done and this_model_num_steps < max_episode_length:
            # Choose action
            actions = model(state)
            action = actions.max(1)[1].data.numpy()
            # Step
            state, reward, done, _ = env.step(action[0])
            returns[i_episode] += reward
            this_model_num_steps += 1
            # Cast state
            state = Variable(torch.from_numpy(state).float(), volatile=True).unsqueeze(0)
    
    mean = np.mean(returns)  # Mean return
    sem = st.sem(returns)    # Standard error of mean
    for conf in [0.9, 0.95, 0.975, 0.99]:
        interval = st.norm.interval(conf, loc=mean, scale=sem)
        half_width = (interval[1] - interval[0])/2
        print("{:2d}% CI = {:5.2f} +/- {:<5.2f},  [{:>5.2f}, {:<5.2f}]".format(int(conf*100), mean, half_width, interval[0], interval[1]))


def supervised_eval(model, train_loader, return_queue, random_seed, silent=False, collect_inputs=False, do_cuda=False, **kwargs):
    """
    Function to evaluate the fitness of a supervised model.

    For supervised training, the training data set loader is viewed as the "environment"
    and is passed in the env variable (train_loader).
    """
    (data, target) = next(iter(train_loader))
    data, target = Variable(data), Variable(target)
    if do_cuda:
        data, target = data.cuda(), target.cuda()
    output = model(data)
    retrn = -F.nll_loss(output, target)
    if do_cuda:
        retrn = retrn.cpu()
    retrn = retrn.data.numpy()[0]
    out = {'seed': random_seed, 'return': retrn, 'n_observations': data.data.size()[0]}
    if collect_inputs:
        # NOTE It is necessary to convert the torch.autograd.Variable to numpy array 
        # in order to correctly transfer this data from the worker thread to the main thread.
        # This is an unfortunate result of how Python pickling handles sending file descriptors.
        # Torch sends tensors via shared memory instead of writing the values to the queue. 
        # The steps are roughly:
        #   1. Background process sends token mp.Queue.
        #   2. When the main process reads the token, it opens a unix socket to the background process.
        #   3. The background process sends the file descriptor via the unix socket.
        out['inputs'] = data.data.numpy()
        # Also print correct prediction ratio
        if not silent:
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct_ratio = pred.eq(target.data.view_as(pred)).sum()/target.data.size()[0]
            print(" | Acc {:4.2f}%".format(correct_ratio*100), end="")
    return_queue.put(out)


def supervised_test(model, test_loader, cuda=False):
    """
    Function to test the performance of a supervised classification model
    """
    model.eval()
    test_loss = 0
    correct = 0
    predictions = []
    targets = []
    for data, target in test_loader:
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        predictions.extend(pred.cpu().numpy().flatten())
        targets.extend(target.cpu().data.numpy().flatten())

    test_loss /= len(test_loader.dataset)
    print('Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    print(confusion_matrix(targets, predictions))
