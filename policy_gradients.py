"""
2018-10-17
Exercise from Arthur Juliani's RL tutorial (adapted for Pytorch)
Part 2: Policy-Based Agents
"""

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

import gym
import pdb

"""
CONTEXT BLOCK:

Cartpole! 
A reward of 1 is received at each timestep if the
pole is still on the cart. :) 

This is now a full RL problem under an MDP.

S: state space 
A: action space
s: current state 
a: selected action 
s': new state 
r = R(s, a): reward 
gamma: decay on future rewards

Experiences are collected in a buffer, then rewards 
are rolled out at once (rollouts / experience traces) 
at the END OF EACH EPISODE.

For environments with sparse or variable rewards, 
aggregating gradients (across rewards in an ep) keeps 
training a bit more stable.

This is REINFORCE / Monte Carlo Policy Gradient!!
"""

def discount_rewards(reward_arr, gamma):

    """this takes in an array of raw rewards from each timestep and
    returns sums of all discounted future rewards.

    input is structured so the most recent reward is the LAST ENTRY."""

    current_r = 0
    d_rewards = []

    for r in reward_arr[::-1]:
        current_r = r + gamma * current_r
        d_rewards.append(current_r)

    return np.array(d_rewards[::-1])


class FriendlyNN(nn.Module):

    """it seems like dropout helps for this task!!"""

    def __init__(self, n_inputs, n_hidden, n_outputs):
        super(FriendlyNN, self).__init__()
        self.fc1 = nn.Linear(n_inputs, n_hidden, bias=False)
        self.drop1 = nn.Dropout(p=0.5)
        self.act1 = nn.ReLU()

        self.fc2 = nn.Linear(n_hidden, n_outputs, bias=False)
        self.act2 = nn.Softmax(dim=-1)

    def forward(self, X):

        X_out = self.fc1(X)
        X_out = self.drop1(X_out)
        X_out = self.act1(X_out)
        X_out = self.fc2(X_out)
        X_out = self.act2(X_out)

        return X_out

def train_a_cartpole():

    env = gym.make('CartPole-v0')
    gamma = 0.99  # discount factor for future rewards

    agent = FriendlyNN(env.observation_space.shape[0],
                       128,
                       env.action_space.n)
    optimizer = torch.optim.Adam(agent.parameters(), lr=0.01)
    max_episodes = 1000

    timesteps = []

    for i in range(max_episodes):

        ep_rewards = []
        state = env.reset()
        done = False

        for timestep in range(1000):
            state = torch.from_numpy(state).float()  # state is an np array of shape (4,)
            action_probs = agent.forward(state)

            """select an action based off the network's probabilities"""
            action = np.random.choice(range(env.action_space.n),
                                      p=action_probs.detach().numpy())
            state, reward, done, _ = env.step(action)
            ep_rewards.append(reward)

            """save the action taken to update the policy!!
            use view(1) to convert from 0-dim tensors to 1-dim (that can be concatenated)"""
            if timestep == 0:
                ep_actions = (-torch.log(action_probs[action])).view(1)
            else:
                ep_actions = torch.cat([ep_actions, (-torch.log(action_probs[action])).view(1)])

            if done:
                break

        """end of episode update!!!"""
        d_rewards = discount_rewards(ep_rewards, gamma)

        """scale rewards to 0 mean 1 var!!!"""
        d_rewards = d_rewards - d_rewards.mean()
        d_rewards = d_rewards / d_rewards.std()
        d_rewards = torch.from_numpy(d_rewards).float()

        """calculate loss!!"""
        loss = torch.sum(torch.mul(ep_actions, d_rewards))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        timesteps.append(timestep)

        if i % 50 == 0:
            print("Episode: {0}, last length: {1}, average length: {2}".format(i,
                                                                               timestep,
                                                                               '{:.2f}'.format(np.mean(timesteps))))

train_a_cartpole()
