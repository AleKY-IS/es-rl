import torch
import torch.nn.functional as F
from torch.autograd import Variable
import time
import IPython
from sklearn.metrics import confusion_matrix


def gym_rollout(args, model, random_seed, return_queue, env, is_antithetic, collect_inputs=False):
    """
    Function to do rollouts of a policy defined by `model` in given environment
    """
    # Reset environment
    # print("Work started")
    # model_time = 0
    state = env.reset()
    state = Variable(torch.from_numpy(state).float(), requires_grad=True).unsqueeze(0)
    retrn = 0
    nsteps = 0
    done = False
    inputs = None
    # Rollout
    while not done and nsteps < args.max_episode_length:
        # Collect states as batch inputs 
        if collect_inputs:
            inputs = torch.cat((inputs, state)) if inputs is not None else state
        # Choose action
        actions = model(state)
        action = actions.max(1)[1].data.numpy()
        # Step
        state, reward, done, _ = env.step(action[0])
        retrn += reward
        nsteps += 1
        # Cast state
        state = Variable(torch.from_numpy(state).float(), requires_grad=True).unsqueeze(0)
    out = {'seed': random_seed, 'return': retrn, 'is_anti': is_antithetic, 'nsteps': nsteps}
    if collect_inputs:
        out['inputs'] = inputs
    return_queue.put(out)


def gym_render(args, model, env):
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
            while not done and this_model_num_steps < args.max_episode_length:
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


def supervised_eval(args, model, random_seed, return_queue, train_loader, is_antithetic, collect_inputs=False):
    """
    Function to evaluate the fitness of a supervised model.

    For supervised training, the training data set loader is viewed as the "environment"
    and is passed in the env variable (train_loader).
    """
    (data, target) = next(iter(train_loader))
    data, target = Variable(data), Variable(target)
    output = model(data)
    pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
    correct_ratio = pred.eq(target.data.view_as(pred)).sum()/target.data.size()[0]
    #reward = -F.nll_loss(output, target)
    #reward.data.numpy()[0]
    out = {'seed': random_seed, 'return': correct_ratio, 'is_anti': is_antithetic, 'nsteps': args.batch_size}
    if collect_inputs:
        out['inputs'] = data
    return_queue.put(out)
    print(correct_ratio)


def supervised_test(args, model, test_loader):
    """
    Function to test the performance of a supervised classification model
    """
    model.eval()
    test_loss = 0
    correct = 0
    predictions = []
    targets = []
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        predictions.extend(pred.numpy().flatten())
        targets.extend(target.data.numpy().flatten())

    test_loss /= len(test_loader.dataset)
    print('Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    print(confusion_matrix(targets, predictions))


    # (2) #I have 2 classes here
    # for ii, data in enumerate(val_dataloader):
    #     input, label = data
    #     val_input = Variable(input, volatile=True).cuda()
    #     val_label = Variable(label.type(t.LongTensor), volatile=True).cuda()
    #     score = model(val_input)
    #     confusion_matrix.add(score.data.squeeze(), label.type(t.LongTensor))
    # print confusion_matrix.conf