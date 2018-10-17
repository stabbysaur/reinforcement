"""
2018-10-08
Exercise from Arthur Juliani's RL tutorial (adapted for Pytorch)
Part 1.5: Contextual bandits!
"""

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

import pdb

"""
CONTEXT BLOCK:

There are now contexts!! 

Each "context" refers to a different bandit. 
Each bandit has 4 arms. 

The NN needs to learn which arm to pull for each bandit!! 
This now has states / actions / rewards but the action taken
does not determine the next state.

Almost at the full RL problem!

Note that this network uses a POLICY GRADIENT approach 
(rather than value-based approaches). The network updates 
towards the correct action, not the value of an action 
in a given state.
"""


class contextual_bandit():

    """taken straight from the blog post :)"""

    def __init__(self):
        self.state = 0
        # List out our bandits. Currently arms 4, 2, and 1 (respectively) are the most optimal.
        self.bandits = np.array([[0.2, 0, -0.0, -5], [0.1, -5, 1, 0.25], [-5, 5, 5, 5]])
        self.num_bandits = self.bandits.shape[0]
        self.num_actions = self.bandits.shape[1]

    def getBandit(self):
        self.state = np.random.randint(0, len(self.bandits))  # Returns a random state for each episode.
        return self.state

    def pullArm(self, action):
        # Get a random number.
        bandit = self.bandits[self.state, action]
        result = np.random.randn(1)
        if result > bandit:
            # return a positive reward.
            return 1
        else:
            # return a negative reward.
            return -1


"""set up NN!"""
class SimpleNN(nn.Module):

    def __init__(self, n_inputs, n_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(n_inputs, n_classes, bias=False)
        # nn_init.uniform_(self.fc1.weight, 0.0, 0.1)

        self.act1 = nn.Sigmoid()

    def forward(self, X):
        output = self.fc1(X)
        output = self.act1(output)

        return output

bandit = contextual_bandit()
agent = SimpleNN(n_inputs=bandit.num_bandits, n_classes=bandit.num_actions)
optimizer = torch.optim.SGD(agent.parameters(), lr=0.05)
episodes = 10000
epsilon = 0.1

rewards = []
for ep in range(episodes):

    """get a bandit!!"""
    band_vector = np.zeros(bandit.num_bandits)
    band = bandit.getBandit()
    band_vector[band] = 1
    band_vector = torch.from_numpy(band_vector).float()

    """pass into agent!!"""
    actions = agent.forward(band_vector)  # this is the current weighting of arms for the given bandit (=state)
    if np.random.rand(1) < epsilon:
        selected = np.random.randint(0, bandit.num_actions - 1)
    else:
        selected = torch.argmax(actions).item()  # pick the best action in the state

    """get reward from taking an action!!!"""
    reward = bandit.pullArm(selected)

    """calculate loss!"""
    loss = -torch.log(actions[selected]) * reward  # same as the non-contextual bandit

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    rewards.append(reward)

    if ep % 100 == 0:
        print("Episode {0}!".format(ep))
        print(sum(rewards) / (ep + 1.))

"""check for whether the agent converged to the right arms for each bandit"""
for band in range(bandit.num_bandits):

    """get a bandit!!"""
    band_vector = np.zeros(bandit.num_bandits)
    band_vector[band] = 1
    band_vector = torch.from_numpy(band_vector).float()

    """pass into agent!!"""
    actions = agent.forward(band_vector)  # this is the current weighting of arms for the given bandit (=state)

    print("The agent thinks action " + str(torch.argmax(actions).item() + 1) + " for bandit " + str(band + 1) + " is the most promising....")
    if torch.argmax(actions).item() == np.argmin(bandit.bandits[band]):
        print("...and it was right!")
    else:
        print("...and it was wrong!")