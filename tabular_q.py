"""
2018-10-04
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


def tabular_q():

    """this function learns the values of each state/action by manually setting up
    a table of all (state, action) pairs and updating the rewards received."""

    env = gym.make('FrozenLake-v0')

    """
    There are 16 possible states (4x4 grid of blocks) and 4 possible actions (directions).
    In total, we need a 16x4 table of Q-values (state, action pairs).
    """

    Q = np.zeros([env.observation_space.n, env.action_space.n])

    """
    Training loop!
    """
    lr = 0.8
    gamma = 0.95
    n_steps = 2000
    sub_steps = 100

    rewards = []
    for i in range(n_steps):
        cur_state = env.reset()  # resets and returns first observation
        cur_reward = 0.  # episode reward

        for j in range(sub_steps):

            """greedily pick the highest-value action with random normal noise"""
            cur_action = np.argmax(Q[cur_state, :] + np.random.randn(1, env.action_space.n) * (1./ (i + 1)))
            next_state, reward, done, _ = env.step(cur_action)

            """update the Q-table with the reward received.
            note that this is structured as the surprise: 
            the old Q-value is shifted by the learning rate * difference in reward received and reward expected."""
            Q[cur_state, cur_action] += lr * (reward + gamma * np.max(Q[next_state, :]) - Q[cur_state, cur_action])

            cur_state = next_state
            cur_reward += reward

            """if done (agent fell into a hole), the episode ends"""
            if done:
                break

        rewards.append(cur_reward)
        print(sum(rewards) / (i + 1.))  # track average reward per episode


tabular_q()
