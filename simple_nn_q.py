"""
2018-10-07
Exercise from Arthur Juliani's RL tutorial (adapted for Pytorch)
Part 0: Q-Learning with Tables and Neural Networks

Update rule (Bellman):
    Q(s, a) = r + gamma * (max(Q(s', a')))
    The expected long-term reward for a given action = immediate reward +
    discounted future reward of the best future action in the next state!
"""

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as nn_init
from torch.autograd import Variable

import pdb

"""
FrozenLake is a 4x4 grid of blocks: 
    --start
    --goal
    --safe
    --dangerous

The agent wishes to navigate start --> goal, moving up/down/left/right.
There is a wind that occasionally blows the agent onto a space they didn't choose.
The reward at each step is 0, goal 1.
"""


class SimpleNN(nn.Module):

    def __init__(self, n_inputs, n_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(n_inputs, n_classes, bias=False)
        # nn_init.uniform_(self.fc1.weight, 0.0, 0.1)

        self.act1 = nn.ReLU()

    def forward(self, X):
        output = self.fc1(X)
        # output = self.act1(output)

        return output


def state_to_tensor(x, l):

    state = np.zeros(l)
    state[x] = 1
    return torch.from_numpy(state).float().unsqueeze(0)


def q_with_nns():

    """
    SOME NOTES ON IMPLEMENTATION:
        -- this was much more of a nightmare than it should have been!!!
        -- this version adds an extra dimension to inputs. it trains properly
           without the extra dim, but in general the first dim given to Pytorch
           models should be batch size (= 1 in this case) and can cause issues in
           other models if dropped.
        -- target q-values should be cloned / detached so they don't affect the cost.
        -- most importantly!!! there is a call of torch.max on the Q-values of
           the new state. THIS GRADIENT SHOULD NOT BE KEPT!!!! use torch.max on
           new_q_values.DATA (or whatever the tensor is called) otherwise the
           training breaks and is REALLY ANNOYING TO DEBUG!!!
        -- wrapping Variables around Tensors doesn't seem to matter (using Pytorch > 0.4).
        -- adding in a negative reward (instead of 0) if the episode ends and the
           agent doesn't reach the goal helps speed up training.
    """

    env = gym.make('FrozenLake-v0')

    """define basic params"""
    lr = 0.01  # learning rate
    episodes = 2000
    steps = 100

    gamma = 0.99
    epsilon = 0.1  # epsilon-greedy param

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    """define model and optimizer!"""
    model = SimpleNN(n_states, n_actions)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    rewards = []
    steps_per_episode = []

    """training loop!"""
    for i in range(episodes):

        state = int(env.reset())
        ep_reward = 0.

        for j in range(steps):

            """
            get the current action.
            with epsilon probability, this will be a random action.
            otherwise greedy.
            """

            state = Variable(state_to_tensor(state, n_states))
            predictions = model.forward(state)

            if np.random.uniform(0.0, 1.0) < epsilon:
                action = env.action_space.sample()
            else:
                action = predictions.argmax(1).item()

            next_state, reward, done, _ = env.step(action)

            if done and (reward == 0):  # add in a negative reward for falling into a hole
                reward = -1.

            """get q-values for the new state!"""
            next_qvals = model.forward(Variable(state_to_tensor(next_state, n_states)))
            max_qval, _ = torch.max(next_qvals.data, 1)  # DO NOT KEEP THIS GRADIENT
            max_qval = torch.FloatTensor(max_qval)

            """update q-value predictions!"""
            q_targets = predictions.clone().detach()
            q_targets[0, action] = reward + torch.mul(max_qval, gamma)

            """calculate loss!"""
            loss = criterion(predictions, q_targets)

            """optimizer / backprop!"""
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            """update episode stats!"""
            state = next_state
            ep_reward += reward

            """check if episode has completed!"""
            if done:
                if reward > 0:
                    epsilon = 1. / ((i / 50) + 10)
                break

        rewards.append(ep_reward)
        steps_per_episode.append(j)

        if i % 100 == 0:
            print("Episode {0}!".format(i))
            print(sum(rewards) / (i + 1.))

    print('\nSuccessful episodes: {}'.format(np.sum(np.array(rewards) > 0.0) / episodes))


q_with_nns()
