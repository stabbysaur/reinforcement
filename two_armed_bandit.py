"""
2018-10-08
Exercise from Arthur Juliani's RL tutorial (adapted for Pytorch)
Part 1: Two-Armed Bandit!
"""

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

import pdb

"""
CONTEXT BLOCK:

There are n (in this case 2!) slot machines with different payout probabilities.

The goal is to learn which slot machine gives the best payout and maximize expected
return by always choosing that machine.

Here the network consists of a set of weights (one for each arm/machine) that represent
how good the agent thinks it is to pull each arm.

To update, we try an arm with an epsilon-greedy policy and shift the weights based on 
the reward received:

    loss = -log(pi) * A
    
where A is the advantage, or how much better the reward was than some baseline. For the
simple bandit case, the baseline used is 0 (so A = reward).

pi is the policy, which corresponds to the chosen action's weight.

If the reward is positive (loss = -log(pi)), then the weight of that arm should be increased.

If the reward is negative (loss = log(pi)), then the weight of that arm should be decreased.
"""

"""set up bandits!"""
bandits = [0.2, 0, -0.2, -5]
epsilon = 0.1  # ep-greedy

def poke_a_bandit(bandit):

    """get a reward from a given bandit!"""

    sample = np.random.randn(1)
    if sample > bandit:
        return 1
    return -1

def simple_bandit():

    weights = torch.nn.Parameter(torch.ones(len(bandits)))
    optimizer = torch.optim.SGD([weights], lr=0.1)  # optimizer takes an iterable of torch params

    episodes = 2000
    rewards = []
    for i in range(episodes):

        """select action -- epsilon-greedy policy"""
        if np.random.uniform(0.0, 1.0) < epsilon:
            choice = np.random.choice(range(len(bandits)))
        else:
            choice = torch.argmax(weights)

        bandit = bandits[choice]
        reward = poke_a_bandit(bandit)

        loss = -torch.log(weights[choice]) * reward

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        rewards.append(reward)
        print('\nAverage reward: {}'.format(np.sum(rewards) / (i + 1.)))
